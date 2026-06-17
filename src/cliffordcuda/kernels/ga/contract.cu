// =============================================================================
// Cl(p, q, r) contraction kernel — CHUNK-ITERATION + WARP-UNIFORM CHUNK SKIP.
// =============================================================================
//
// Used for both left and right contraction. The operation is encoded purely
// via the packed_valid LUT: bit t of packed_valid[k, c] is 1 iff the (i, j)
// term contributes to c[k] under the chosen operation AND sigma_val(i, j) != 0.
//
// Chunk-skip: per output blade k, when packed_valid[k, c] == 0 every lane
// in that chunk would multiply by zero. The kernel skips the whole chunk via
// a warp-uniform branch on the staged valid word — no operand SMEM access,
// no FMAs, no sign-bit extract. This captures (a) chunks where the operation
// predicate (e.g. (i & k) == 0 for left, (k & ~i) == 0 for right) excludes
// every lane, and (b) chunks where every term hits a degenerate generator.
//
// The kernel binary is shared across left and right contraction; the only
// thing that changes is what packed_valid encodes. (See left_contract.py and
// right_contract.py for the two LUT builders.)
//
// Layout otherwise mirrors geom_prod.cu's HAS_ZEROS=true path:
//   * cp.async operand prelude (Ampere+).
//   * SMEM-staged sign + valid rows.
//   * K=4 inner-loop unroll.
//   * W warps per block.
//   * XOR-mod-32 bank-conflict-free SMEM access.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_pipeline.h>
#include <c10/cuda/CUDAGuard.h>


template<int N, int W, int K>
__global__ void contract_fwd_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int*   __restrict__ packed_sign,
    const int*   __restrict__ packed_valid,
    float*       __restrict__ c,
    int batch)
{
    constexpr int DIM    = 1 << N;
    constexpr int CHUNKS = DIM >> 5;
    constexpr int N4     = DIM >> 2;

    int b_idx = blockIdx.x;
    if (b_idx >= batch) return;
    int output_base = blockIdx.y * W;

    extern __shared__ unsigned char smem_raw[];
    float* sA   = reinterpret_cast<float*>(smem_raw);
    float* sB   = sA + DIM;
    int*   sSig = reinterpret_cast<int*>(sB + DIM);             // W * CHUNKS int32s
    int*   sVal = sSig + W * CHUNKS;                            // W * CHUNKS int32s

    int tid = threadIdx.x;
    int blocksize = W * 32;

    // Async operand prelude.
    const float4* a4  = reinterpret_cast<const float4*>(a + b_idx * DIM);
    const float4* bb4 = reinterpret_cast<const float4*>(b + b_idx * DIM);
    float4* sA4 = reinterpret_cast<float4*>(sA);
    float4* sB4 = reinterpret_cast<float4*>(sB);
    #pragma unroll 1
    for (int i = tid; i < N4; i += blocksize) {
        __pipeline_memcpy_async(&sA4[i], &a4[i],  16);
        __pipeline_memcpy_async(&sB4[i], &bb4[i], 16);
    }
    __pipeline_commit();

    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int k       = output_base + warp_id;

    // Stage sign + validity rows for this warp.
    if (k < DIM) {
        int* sSig_warp = sSig + warp_id * CHUNKS;
        int* sVal_warp = sVal + warp_id * CHUNKS;
        for (int c_idx = lane; c_idx < CHUNKS; c_idx += 32) {
            sSig_warp[c_idx] = packed_sign [k * CHUNKS + c_idx];
            sVal_warp[c_idx] = packed_valid[k * CHUNKS + c_idx];
        }
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    if (k >= DIM) return;

    int* sSig_warp = sSig + warp_id * CHUNKS;
    int* sVal_warp = sVal + warp_id * CHUNKS;

    float acc = 0.0f;
    int c_idx = 0;

    // K-unrolled main loop.
    #pragma unroll 1
    for (; c_idx + K <= CHUNKS; c_idx += K) {
        float local = 0.0f;
        #pragma unroll
        for (int u = 0; u < K; u++) {
            int vword = sVal_warp[c_idx + u];
            if (vword == 0) continue;             // warp-uniform: every lane skips.
            int chunk = (c_idx + u) << 5;
            int i = chunk + lane;
            int j = i ^ k;
            int sword = sSig_warp[c_idx + u];
            float s = 1.0f - 2.0f * float((sword >> lane) & 1);
            float v =                float((vword >> lane) & 1);
            local += v * s * sA[i] * sB[j];
        }
        acc += local;
    }
    // Remainder.
    #pragma unroll 1
    for (; c_idx < CHUNKS; c_idx++) {
        int vword = sVal_warp[c_idx];
        if (vword == 0) continue;
        int chunk = c_idx << 5;
        int i = chunk + lane;
        int j = i ^ k;
        int sword = sSig_warp[c_idx];
        float s = 1.0f - 2.0f * float((sword >> lane) & 1);
        float v =                float((vword >> lane) & 1);
        acc += v * s * sA[i] * sB[j];
    }

    // Warp reduce via XOR butterfly.
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, delta);

    if (lane == 0)
        c[b_idx * DIM + k] = acc;
}


#define LAUNCH(N_VAL, W_VAL, K_VAL)                                                    \
    do {                                                                                \
        constexpr int DIM = 1 << (N_VAL);                                               \
        constexpr int CHUNKS = DIM >> 5;                                                \
        int blocks_y = (DIM + (W_VAL) - 1) / (W_VAL);                                   \
        dim3 grid(batch, blocks_y);                                                     \
        int threads = (W_VAL) * 32;                                                     \
        int sig_smem = (W_VAL) * CHUNKS * (int)sizeof(int);                             \
        int smem = (2 * DIM * (int)sizeof(float)) + 2 * sig_smem;                       \
        if (smem > 48 * 1024)                                                           \
            TORCH_CHECK(cudaFuncSetAttribute(                                                       \
                contract_fwd_kernel<(N_VAL), (W_VAL), (K_VAL)>,                         \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "dynamic SMEM opt-in failed");                     \
        contract_fwd_kernel<(N_VAL), (W_VAL), (K_VAL)>                                  \
            <<<grid, threads, smem, stream>>>(a_ptr, b_ptr, sign_ptr, valid_ptr,        \
                                              c_ptr, batch);                            \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } while (0)


torch::Tensor contract_fwd(torch::Tensor a, torch::Tensor b,
                           torch::Tensor packed_sign,
                           torch::Tensor packed_valid) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed_sign .is_cuda() && packed_sign .is_contiguous()
                && packed_sign .scalar_type() == torch::kInt32);
    TORCH_CHECK(packed_valid.is_cuda() && packed_valid.is_contiguous()
                && packed_valid.scalar_type() == torch::kInt32);
    TORCH_CHECK(a.sizes() == b.sizes()); TORCH_CHECK(a.dim() == 2);
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    int dim = a.size(1);
    int N = 0;
    while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim, "dim must be a power of two");
    TORCH_CHECK(N >= 5, "N>=5 required");
    TORCH_CHECK(packed_sign .size(0) == dim && packed_sign .size(1) == dim / 32);
    TORCH_CHECK(packed_valid.size(0) == dim && packed_valid.size(1) == dim / 32);

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* a_ptr     = a.data_ptr<float>();
    const float* b_ptr     = b.data_ptr<float>();
    const int*   sign_ptr  = packed_sign .data_ptr<int>();
    const int*   valid_ptr = packed_valid.data_ptr<int>();
    float*       c_ptr     = c.data_ptr<float>();

    if      (N == 5)  LAUNCH(5,  4, 2);
    else if (N == 6)  LAUNCH(6,  4, 2);
    else if (N == 7)  LAUNCH(7,  4, 2);
    else if (N == 8)  LAUNCH(8,  4, 2);
    else if (N == 9)  LAUNCH(9,  16, 4);
    else if (N == 10) LAUNCH(10, 16, 4);
    else if (N == 11) LAUNCH(11, 16, 4);
    else if (N == 12) LAUNCH(12, 16, 4);
    else if (N == 13) LAUNCH(13, 16, 4);
    else TORCH_CHECK(false, "unsupported N (5..13)");

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("contract_fwd", &contract_fwd,
          "Cl(p, q, r) contraction (left/right via packed_valid LUT, chunk-skip)",
          pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("packed_sign"), pybind11::arg("packed_valid"));
}
