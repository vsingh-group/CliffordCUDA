// =============================================================================
// Cl(p, q, r) inner (Hestenes) product kernel — CHUNK-ITERATION + LANE PREDICATE.
// =============================================================================
//
// Inner product: e_i · e_j is non-zero iff i ⊆ j OR j ⊆ i (one is a subset
// of the other in bit-pattern indexing). Equivalently, with j = i ^ k for
// output blade k:
//   (k ⊆ i)        <=> (k & ~i) == 0    (case j ⊆ i)
//   (i & k == 0)                        (case i ⊆ j)
//
// This is the geom_prod kernel with one extra per-lane predicate. For
// Cl(p, q, r) with r > 0, an optional packed_valid LUT additionally masks
// terms where shared bases hit a degenerate generator. The HAS_ZEROS=false
// path is bit-for-bit identical to the prior Cl(n, 0) kernel.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_pipeline.h>
#include <c10/cuda/CUDAGuard.h>


template<int N, int W, int K, bool HAS_ZEROS, bool USE_SKIP>
__global__ void inner_prod_fwd_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int*   __restrict__ packed_sign,
    const int*   __restrict__ packed_valid,
    float*       __restrict__ c,
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
    int*   sVal = HAS_ZEROS ? (sSig + W * CHUNKS) : nullptr;

    int tid = threadIdx.x;
    int blocksize = W * 32;

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
        if constexpr (HAS_ZEROS) {
            int* sVal_warp = sVal + warp_id * CHUNKS;
            for (int c_idx = lane; c_idx < CHUNKS; c_idx += 32) {
                sVal_warp[c_idx] = packed_valid[k * CHUNKS + c_idx];
            }
        }
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    if (k >= DIM) return;

    int* sSig_warp = sSig + warp_id * CHUNKS;
    int* sVal_warp = HAS_ZEROS ? (sVal + warp_id * CHUNKS) : nullptr;

    // Predicate-aware chunk skip (USE_SKIP=true). The Inner predicate
    // (k ⊆ i) OR (i & k == 0) holds for SOME lane in a chunk iff
    // (k_high ⊆ chunk) OR (chunk & k_high == 0), where k_high = k & ~31.
    // Skip iff neither: (chunk & k_high) ∉ {0, k_high}. Address-only — no
    // SMEM load, just a bit test on the chunk address.
    int k_high = k & ~31;

    float acc = 0.0f;
    int c_idx = 0;
    #pragma unroll 1
    for (; c_idx + K <= CHUNKS; c_idx += K) {
        float local = 0.0f;
        #pragma unroll
        for (int u = 0; u < K; u++) {
            int chunk = (c_idx + u) << 5;
            if constexpr (USE_SKIP) {
                int x = chunk & k_high;
                if (x != 0 && x != k_high) continue;
            }
            int i = chunk + lane;
            int j = i ^ k;
            int word = sSig_warp[c_idx + u];
            float s = 1.0f - 2.0f * float((word >> lane) & 1);
            // Inner predicate: (k ⊆ i) OR (i & k == 0).
            float valid = float(((k & ~i) == 0) | ((i & k) == 0));
            if constexpr (HAS_ZEROS) {
                int vword = sVal_warp[c_idx + u];
                float v = float((vword >> lane) & 1);
                local += v * valid * s * sA[i] * sB[j];
            } else {
                local += valid * s * sA[i] * sB[j];
            }
        }
        acc += local;
    }
    #pragma unroll 1
    for (; c_idx < CHUNKS; c_idx++) {
        int chunk = c_idx << 5;
        if constexpr (USE_SKIP) {
            int x = chunk & k_high;
            if (x != 0 && x != k_high) continue;
        }
        int i = chunk + lane;
        int j = i ^ k;
        int word = sSig_warp[c_idx];
        float s = 1.0f - 2.0f * float((word >> lane) & 1);
        float valid = float(((k & ~i) == 0) | ((i & k) == 0));
        if constexpr (HAS_ZEROS) {
            int vword = sVal_warp[c_idx];
            float v = float((vword >> lane) & 1);
            acc += v * valid * s * sA[i] * sB[j];
        } else {
            acc += valid * s * sA[i] * sB[j];
        }
    }

    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, delta);

    if (lane == 0)
        c[b_idx * DIM + k] = acc;
}


#define LAUNCH_INNER(N_VAL, W_VAL, K_VAL, HAS_Z, USE_S)                                          \
    do {                                                                                          \
        constexpr int DIM = 1 << (N_VAL);                                                         \
        constexpr int CHUNKS = DIM >> 5;                                                          \
        int blocks_y = (DIM + (W_VAL) - 1) / (W_VAL);                                             \
        dim3 grid(batch, blocks_y);                                                               \
        int threads = (W_VAL) * 32;                                                               \
        int sig_smem = (W_VAL) * CHUNKS * (int)sizeof(int);                                       \
        int smem = (2 * DIM * (int)sizeof(float))                                                 \
                 + sig_smem                                                                       \
                 + ((HAS_Z) ? sig_smem : 0);                                                      \
        if (smem > 48 * 1024)                                                                     \
            TORCH_CHECK(cudaFuncSetAttribute(                                                                 \
                inner_prod_fwd_kernel<(N_VAL), (W_VAL), (K_VAL), (HAS_Z), (USE_S)>,               \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "dynamic SMEM opt-in failed");                               \
        inner_prod_fwd_kernel<(N_VAL), (W_VAL), (K_VAL), (HAS_Z), (USE_S)>                        \
            <<<grid, threads, smem, stream>>>(a_ptr, b_ptr, sign_ptr, valid_ptr,                  \
                                              c_ptr, batch);                                      \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } while (0)


torch::Tensor inner_prod_fwd(torch::Tensor a, torch::Tensor b,
                             torch::Tensor packed_sign,
                             c10::optional<torch::Tensor> packed_valid_opt) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed_sign.is_cuda() && packed_sign.is_contiguous()
                && packed_sign.scalar_type() == torch::kInt32);
    TORCH_CHECK(a.sizes() == b.sizes()); TORCH_CHECK(a.dim() == 2);
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    int dim = a.size(1);
    int N = 0;
    while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim);
    TORCH_CHECK(N >= 5);
    TORCH_CHECK(packed_sign.size(0) == dim && packed_sign.size(1) == dim / 32);

    bool has_zeros = packed_valid_opt.has_value();
    const int* valid_ptr = nullptr;
    if (has_zeros) {
        auto& pv = packed_valid_opt.value();
        TORCH_CHECK(pv.is_cuda() && pv.is_contiguous() && pv.scalar_type() == torch::kInt32);
        TORCH_CHECK(pv.size(0) == dim && pv.size(1) == dim / 32);
        valid_ptr = pv.data_ptr<int>();
    }

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* a_ptr     = a.data_ptr<float>();
    const float* b_ptr     = b.data_ptr<float>();
    const int*   sign_ptr  = packed_sign.data_ptr<int>();
    float*       c_ptr     = c.data_ptr<float>();

    if (!has_zeros) {
        if      (N == 5)  LAUNCH_INNER(5,  4, 2, false, false);
        else if (N == 6)  LAUNCH_INNER(6,  4, 2, false, false);
        else if (N == 7)  LAUNCH_INNER(7,  4, 2, false, false);
        else if (N == 8)  LAUNCH_INNER(8,  4, 2, false, false);
        else if (N == 9)  LAUNCH_INNER(9,  16, 4, false, false);
        else if (N == 10) LAUNCH_INNER(10, 16, 4, false, false);
        else if (N == 11) LAUNCH_INNER(11, 16, 4, false, false);
        else if (N == 12) LAUNCH_INNER(12, 16, 4, false, false);
        else if (N == 13) LAUNCH_INNER(13, 16, 4, false, false);
        else TORCH_CHECK(false, "unsupported N");
    } else {
        if      (N == 5)  LAUNCH_INNER(5,  4, 2, true, false);
        else if (N == 6)  LAUNCH_INNER(6,  4, 2, true, false);
        else if (N == 7)  LAUNCH_INNER(7,  4, 2, true, false);
        else if (N == 8)  LAUNCH_INNER(8,  4, 2, true, false);
        else if (N == 9)  LAUNCH_INNER(9,  16, 4, true, false);
        else if (N == 10) LAUNCH_INNER(10, 16, 4, true, false);
        else if (N == 11) LAUNCH_INNER(11, 16, 4, true, false);
        else if (N == 12) LAUNCH_INNER(12, 16, 4, true, false);
        else if (N == 13) LAUNCH_INNER(13, 16, 4, true, false);
        else TORCH_CHECK(false, "unsupported N");
    }

    return c;
}


// `inner_prod_fwd_skip`: same kernel but with the predicate-aware chunk-skip
// enabled. Address-only — no SMEM load needed for the decision. Skips chunks
// where no lane could possibly satisfy (k ⊆ i) OR (i & k == 0).
torch::Tensor inner_prod_fwd_skip(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor packed_sign,
                                  c10::optional<torch::Tensor> packed_valid_opt) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed_sign.is_cuda() && packed_sign.is_contiguous()
                && packed_sign.scalar_type() == torch::kInt32);
    TORCH_CHECK(a.sizes() == b.sizes()); TORCH_CHECK(a.dim() == 2);
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    int dim = a.size(1);
    int N = 0;
    while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim);
    TORCH_CHECK(N >= 5);
    TORCH_CHECK(packed_sign.size(0) == dim && packed_sign.size(1) == dim / 32);

    bool has_zeros = packed_valid_opt.has_value();
    const int* valid_ptr = nullptr;
    if (has_zeros) {
        auto& pv = packed_valid_opt.value();
        TORCH_CHECK(pv.is_cuda() && pv.is_contiguous() && pv.scalar_type() == torch::kInt32);
        TORCH_CHECK(pv.size(0) == dim && pv.size(1) == dim / 32);
        valid_ptr = pv.data_ptr<int>();
    }

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* a_ptr     = a.data_ptr<float>();
    const float* b_ptr     = b.data_ptr<float>();
    const int*   sign_ptr  = packed_sign.data_ptr<int>();
    float*       c_ptr     = c.data_ptr<float>();

    if (!has_zeros) {
        if      (N == 5)  LAUNCH_INNER(5,  4, 2, false, true);
        else if (N == 6)  LAUNCH_INNER(6,  4, 2, false, true);
        else if (N == 7)  LAUNCH_INNER(7,  4, 2, false, true);
        else if (N == 8)  LAUNCH_INNER(8,  4, 2, false, true);
        else if (N == 9)  LAUNCH_INNER(9,  16, 4, false, true);
        else if (N == 10) LAUNCH_INNER(10, 16, 4, false, true);
        else if (N == 11) LAUNCH_INNER(11, 16, 4, false, true);
        else if (N == 12) LAUNCH_INNER(12, 16, 4, false, true);
        else if (N == 13) LAUNCH_INNER(13, 16, 4, false, true);
        else TORCH_CHECK(false, "unsupported N");
    } else {
        if      (N == 5)  LAUNCH_INNER(5,  4, 2, true, true);
        else if (N == 6)  LAUNCH_INNER(6,  4, 2, true, true);
        else if (N == 7)  LAUNCH_INNER(7,  4, 2, true, true);
        else if (N == 8)  LAUNCH_INNER(8,  4, 2, true, true);
        else if (N == 9)  LAUNCH_INNER(9,  16, 4, true, true);
        else if (N == 10) LAUNCH_INNER(10, 16, 4, true, true);
        else if (N == 11) LAUNCH_INNER(11, 16, 4, true, true);
        else if (N == 12) LAUNCH_INNER(12, 16, 4, true, true);
        else if (N == 13) LAUNCH_INNER(13, 16, 4, true, true);
        else TORCH_CHECK(false, "unsupported N");
    }

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inner_prod_fwd", &inner_prod_fwd,
          "Cl(p, q, r) inner product (Hestenes), forward",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"),
          pybind11::arg("packed_valid") = c10::optional<torch::Tensor>());
    m.def("inner_prod_fwd_skip", &inner_prod_fwd_skip,
          "Inner (Hestenes) forward with predicate-aware address-only chunk skip",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"),
          pybind11::arg("packed_valid") = c10::optional<torch::Tensor>());
}
