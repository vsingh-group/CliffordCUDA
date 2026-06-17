// =============================================================================
// Cl(n, 0) wedge (exterior) product kernel — CHUNK-ITERATION VARIANT.
// =============================================================================
//
// One of two wedge implementations in this project. See wedge_prod_subset.cu
// for the SUBSET-ITERATION variant.
//
// HOW THESE TWO VARIANTS DIFFER
// -----------------------------
// Both compute the same thing:
//   c[k] = sum_{i ⊆ k} sigma(i, i^k) * a[i] * b[i^k]    (bit-pattern indexing)
// where i ⊆ k <=> (i & ~k) == 0; for i ⊄ k the term is zero by the wedge mask.
//
// CHUNK variant (this file):
//   - Each warp k iterates the FULL i-space in chunks of 32.
//   - Skips a whole chunk only when (chunk & ~k) != 0 (warp-uniform branch).
//   - Within a "valid" chunk, all 32 lanes still issue SMEM loads; lanes
//     where i ⊄ k get zeroed via a per-lane `valid` multiplier.
//   - Reuses geom_prod's packed_sign LUT verbatim (sigma is the same).
//   - Inherits all geom_prod optimizations: cp.async operand prelude,
//     SMEM-staged sign row, K=4 inner unroll, XOR-mod-32 bank-conflict-free
//     SMEM access, warp-shuffle reduce.
//   - Faster when batch is small (no LUT-load overhead) — wins at batch=1 at
//     all n.
//
// SUBSET variant (wedge_prod_subset.cu):
//   - Each warp k enumerates ONLY the 2^|k| valid subsets via a precomputed
//     LUT. Lane t in warp iter w handles the t-th subset.
//   - No wasted SMEM loads for |k| >= 5 — every lane works on a real term.
//   - Adds a global LUT load per warp iter; bit-deposit access pattern is
//     not bank-conflict-free.
//   - Optionally: warps in a block share the same |k| (grade ordering) so
//     n_iters is uniform per block and warps don't idle for slow-warp
//     completion.
//   - Faster than the chunk variant at batch ≥ 256 and n ≥ 8 (up to ~2.8×
//     at n=12 batch=256 with grade ordering).
//
// WHEN EACH IS FASTER (empirical, A100, fp32)
// -------------------------------------------
//   batch=1:                chunk wins everywhere (kernel-launch dominated)
//   batch=256, n>=9:        subset+grade-order wins (1.4–2.8× over chunk)
//   batch=4096, n in 8..10: subset wins (~1.5×)
//   batch=4096, n>=11:      subset (without grade order) wins (~1.5–1.6×)
//
// IMPLEMENTATION NOTES (this file — chunk variant)
// ------------------------------------------------
// Wedge-specific changes vs geom_prod's chunk kernel:
//   * Warp-uniform chunk skip: `if ((chunk & ~k) != 0) continue;` skips
//     entire chunks where no lane has i ⊆ k. Realizes the (4/3)^n speedup
//     at the chunk level.
//   * Lane-level predicate inside valid chunks: `valid = ((i & ~k) == 0)`.
//     Multiply into the FMA so invalid lanes contribute 0. (SMEM bandwidth
//     not saved; only the FMAs.)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_pipeline.h>
#include <c10/cuda/CUDAGuard.h>


template<int N, int W, int K>
__global__ void wedge_prod_fwd_kernel(
    const float*  __restrict__ a,
    const float*  __restrict__ b,
    const int*    __restrict__ packed_sign,
    float*        __restrict__ c,
    int batch)
{
    constexpr int DIM = 1 << N;
    constexpr int CHUNKS = DIM >> 5;
    constexpr int N4 = DIM >> 2;

    int b_idx = blockIdx.x;
    if (b_idx >= batch) return;
    int output_base = blockIdx.y * W;

    extern __shared__ unsigned char smem_raw[];
    float* sA   = reinterpret_cast<float*>(smem_raw);
    float* sB   = sA + DIM;
    int*   sSig = reinterpret_cast<int*>(sB + DIM);

    int tid = threadIdx.x;
    int blocksize = W * 32;

    // Operand cp.async.
    const float4* a4  = reinterpret_cast<const float4*>(a + b_idx * DIM);
    const float4* bb4 = reinterpret_cast<const float4*>(b + b_idx * DIM);
    float4* sA4 = reinterpret_cast<float4*>(sA);
    float4* sB4 = reinterpret_cast<float4*>(sB);
    #pragma unroll 1
    for (int i = tid; i < N4; i += blocksize) {
        __pipeline_memcpy_async(&sA4[i], &a4[i], 16);
        __pipeline_memcpy_async(&sB4[i], &bb4[i], 16);
    }
    __pipeline_commit();

    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int k       = output_base + warp_id;

    if (k < DIM) {
        int* sSig_warp = sSig + warp_id * CHUNKS;
        for (int c_idx = lane; c_idx < CHUNKS; c_idx += 32) {
            sSig_warp[c_idx] = packed_sign[k * CHUNKS + c_idx];
        }
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    if (k >= DIM) return;

    int* sSig_warp = sSig + warp_id * CHUNKS;
    int  k_mask = ~k;     // bits NOT set in k

    float acc = 0.0f;

    int c_idx = 0;
    #pragma unroll 1
    for (; c_idx + K <= CHUNKS; c_idx += K) {
        float local = 0.0f;
        #pragma unroll
        for (int u = 0; u < K; u++) {
            int chunk = (c_idx + u) << 5;
            // Warp-uniform chunk skip: chunk's high bits must be a subset of k.
            // (Lane low-bit check is below; this gates the whole 32-lane chunk.)
            if ((chunk & k_mask) != 0) continue;
            int i = chunk + lane;
            int j = i ^ k;
            int word = sSig_warp[c_idx + u];
            float s = 1.0f - 2.0f * float((word >> lane) & 1);
            // Lane-level predicate: i ⊆ k <=> (i & ~k) == 0.
            float valid = float((i & k_mask) == 0);
            local += valid * s * sA[i] * sB[j];
        }
        acc += local;
    }
    // Remainder.
    #pragma unroll 1
    for (; c_idx < CHUNKS; c_idx++) {
        int chunk = c_idx << 5;
        if ((chunk & k_mask) != 0) continue;
        int i = chunk + lane;
        int j = i ^ k;
        int word = sSig_warp[c_idx];
        float s = 1.0f - 2.0f * float((word >> lane) & 1);
        float valid = float((i & k_mask) == 0);
        acc += valid * s * sA[i] * sB[j];
    }

    // Warp reduce.
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, delta);

    if (lane == 0)
        c[b_idx * DIM + k] = acc;
}


#define LAUNCH_WEDGE(N_VAL, W_VAL, K_VAL)                                              \
    do {                                                                                \
        constexpr int DIM = 1 << (N_VAL);                                               \
        constexpr int CHUNKS = DIM >> 5;                                                \
        int blocks_y = (DIM + (W_VAL) - 1) / (W_VAL);                                   \
        dim3 grid(batch, blocks_y);                                                     \
        int threads = (W_VAL) * 32;                                                     \
        int smem = (2 * DIM * (int)sizeof(float))                                       \
                 + ((W_VAL) * CHUNKS * (int)sizeof(int));                               \
        if (smem > 48 * 1024)                                                           \
            TORCH_CHECK(cudaFuncSetAttribute(                                                       \
                wedge_prod_fwd_kernel<(N_VAL), (W_VAL), (K_VAL)>,                       \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "dynamic SMEM opt-in failed");                     \
        wedge_prod_fwd_kernel<(N_VAL), (W_VAL), (K_VAL)>                                \
            <<<grid, threads, smem, stream>>>(a_ptr, b_ptr, sign_ptr, c_ptr, batch);    \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } while (0)


torch::Tensor wedge_prod_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor packed_sign) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed_sign.is_cuda() && packed_sign.is_contiguous()
                && packed_sign.scalar_type() == torch::kInt32);
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dim() == 2);
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    int dim = a.size(1);
    int N = 0;
    while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim, "dim must be a power of two");
    TORCH_CHECK(N >= 5, "N>=5 required");
    TORCH_CHECK(packed_sign.size(0) == dim && packed_sign.size(1) == dim / 32);

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* a_ptr     = a.data_ptr<float>();
    const float* b_ptr     = b.data_ptr<float>();
    const int*   sign_ptr  = packed_sign.data_ptr<int>();
    float*       c_ptr     = c.data_ptr<float>();

    if      (N == 5)  LAUNCH_WEDGE(5,  4, 2);
    else if (N == 6)  LAUNCH_WEDGE(6,  4, 2);
    else if (N == 7)  LAUNCH_WEDGE(7,  4, 2);
    else if (N == 8)  LAUNCH_WEDGE(8,  4, 2);
    else if (N == 9)  LAUNCH_WEDGE(9,  16, 4);
    else if (N == 10) LAUNCH_WEDGE(10, 16, 4);
    else if (N == 11) LAUNCH_WEDGE(11, 16, 4);
    else if (N == 12) LAUNCH_WEDGE(12, 16, 4);
    else if (N == 13) LAUNCH_WEDGE(13, 16, 4);
    else TORCH_CHECK(false, "unsupported N (5..13)");

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wedge_prod_fwd", &wedge_prod_fwd,
          "Cl(n, 0) wedge product, forward",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"));
}
