// General Cl(p, q, r) geometric product kernel.
//
// Convention: bit-pattern blade indexing. Blade index i in [0, 2^n) is the
// subset whose generator k is included iff bit k of i is 1. Then
//   e_i * e_j = sigma_val(i, j) * e_{i XOR j}
// where sigma_val(i, j) = (-1)^reorder_parity * prod_{k in i & j} metric[k].
// reorder_parity = |{(a, b) : a > b, bit_a(i)=1, bit_b(j)=1}|.
//
// Two LUTs encode the signature:
//   packed_sign[k, c]:  bit t = 1 iff sigma_val(c*32+t, (c*32+t)^k) == -1
//   packed_valid[k, c]: bit t = 1 iff sigma_val(c*32+t, (c*32+t)^k) != 0
// For Cl(p, q) (non-degenerate), every term is ±1 and packed_valid is omitted
// (HAS_ZEROS=false). For Cl(p, q, r) with r > 0, some terms vanish and
// packed_valid masks them at the lane level. The HAS_ZEROS=false path is
// bit-for-bit identical to the prior Cl(n, 0) kernel.
//
// Output:
//   c[k] = sum_i sigma_val(i, i^k) * a[i] * b[i^k]   for k in [0, 2^n).
//
// Each warp produces ONE output blade k for ONE batch element. The 32 lanes
// each accumulate dim/32 partial products and warp-reduce via __shfl_xor_sync.
// One block handles one batch element with W warps; the y-axis of the grid
// stripes the dim outputs across blocks.
//
// SMEM access pattern (bank-conflict-free at warp granularity by construction):
//   sA[chunk + lane]              -> bank = lane                 (32 distinct)
//   sB[(chunk + lane) ^ k]        -> bank = lane ^ (k & 31)      (32 distinct)
// when chunk is a multiple of 32. The XOR-mod-32 algebra makes the access
// pattern a permutation of bank indices for any k.
//
// Optimizations applied:
//   * Packed sign LUT (32 signs per int32) input.
//   * Bank-conflict-free SMEM access by XOR-mod-32 algebra (no perm search).
//   * SMEM staging of operands a, b via cp.async global -> SMEM (Ampere+).
//     The operand prelude overlaps with the synchronous sign-row staging.
//   * SMEM staging of the warp's sign row.
//   * K=4 inner-loop unroll.
//   * W warps per block: more outputs per block amortize the operand load.
//   * `cudaFuncSetAttribute` to opt into >48 KB SMEM at large n.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_pipeline.h>      // cp.async (Ampere+)
#include <c10/cuda/CUDAGuard.h>


template<int N, int W, int K, bool HAS_ZEROS, bool USE_SKIP>
__global__ void geom_prod_fwd_kernel(
    const float*  __restrict__ a,            // (B, dim)
    const float*  __restrict__ b,            // (B, dim)
    const int*    __restrict__ packed_sign,  // (dim, dim/32) int32
    const int*    __restrict__ packed_valid, // (dim, dim/32) int32, used iff HAS_ZEROS
    float*        __restrict__ c,            // (B, dim)
    int batch)
{
    constexpr int DIM = 1 << N;
    constexpr int CHUNKS = DIM >> 5;       // dim / 32
    constexpr int N4 = DIM >> 2;           // dim / 4 (number of float4s)

    int b_idx = blockIdx.x;
    if (b_idx >= batch) return;
    int output_base = blockIdx.y * W;

    extern __shared__ unsigned char smem_raw[];
    float* sA   = reinterpret_cast<float*>(smem_raw);
    float* sB   = sA + DIM;
    int*   sSig = reinterpret_cast<int*>(sB + DIM);     // W * CHUNKS int32s
    int*   sVal = HAS_ZEROS ? (sSig + W * CHUNKS) : nullptr;

    int tid = threadIdx.x;
    int blocksize = W * 32;

    // Async global -> SMEM operand prelude, 16 bytes per issue.
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

    // Stage this warp's sign (and valid) row into SMEM.
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

    float acc = 0.0f;

    // K-unrolled main loop.
    int c_idx = 0;
    #pragma unroll 1
    for (; c_idx + K <= CHUNKS; c_idx += K) {
        float local = 0.0f;
        #pragma unroll
        for (int u = 0; u < K; u++) {
            int chunk = (c_idx + u) << 5;
            int i = chunk + lane;
            int j = i ^ k;
            int word = sSig_warp[c_idx + u];
            float s = 1.0f - 2.0f * float((word >> lane) & 1);
            if constexpr (HAS_ZEROS) {
                int vword = sVal_warp[c_idx + u];
                if constexpr (USE_SKIP) {
                    // Warp-uniform chunk-skip when every lane's term is zero.
                    if (vword == 0) continue;
                }
                float v = float((vword >> lane) & 1);
                local += v * s * sA[i] * sB[j];
            } else {
                local += s * sA[i] * sB[j];
            }
        }
        acc += local;
    }
    // Remainder.
    #pragma unroll 1
    for (; c_idx < CHUNKS; c_idx++) {
        int chunk = c_idx << 5;
        int i = chunk + lane;
        int j = i ^ k;
        int word = sSig_warp[c_idx];
        float s = 1.0f - 2.0f * float((word >> lane) & 1);
        if constexpr (HAS_ZEROS) {
            int vword = sVal_warp[c_idx];
            if constexpr (USE_SKIP) {
                if (vword == 0) continue;
            }
            float v = float((vword >> lane) & 1);
            acc += v * s * sA[i] * sB[j];
        } else {
            acc += s * sA[i] * sB[j];
        }
    }

    // Warp reduce via XOR butterfly.
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, delta);

    if (lane == 0)
        c[b_idx * DIM + k] = acc;
}


#define LAUNCH(N_VAL, W_VAL, K_VAL, HAS_Z, USE_S)                                       \
    do {                                                                                 \
        constexpr int DIM = 1 << (N_VAL);                                                \
        constexpr int CHUNKS = DIM >> 5;                                                 \
        int blocks_y = (DIM + (W_VAL) - 1) / (W_VAL);                                    \
        dim3 grid(batch, blocks_y);                                                      \
        int threads = (W_VAL) * 32;                                                      \
        int sig_smem = (W_VAL) * CHUNKS * (int)sizeof(int);                              \
        int smem = (2 * DIM * (int)sizeof(float))                                        \
                 + sig_smem                                                              \
                 + ((HAS_Z) ? sig_smem : 0);                                             \
        if (smem > 48 * 1024)                                                            \
            TORCH_CHECK(cudaFuncSetAttribute(                                            \
                geom_prod_fwd_kernel<(N_VAL), (W_VAL), (K_VAL), (HAS_Z), (USE_S)>,       \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,       \
                "geom_prod: dynamic SMEM opt-in failed (need ", smem, " bytes)");        \
        geom_prod_fwd_kernel<(N_VAL), (W_VAL), (K_VAL), (HAS_Z), (USE_S)>                \
            <<<grid, threads, smem, stream>>>(a_ptr, b_ptr, sign_ptr, valid_ptr,         \
                                              c_ptr, batch);                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                  \
    } while (0)


// Single-k GP forward. `use_skip` enables the warp-uniform chunk-skip on the
// HAS_ZEROS path (helps when σ has many fully-zero chunks — wedge/lc/rc/
// regressive backward LUTs; hurts on denser σ, e.g. inner backward regresses
// ~25% at small n). The HAS_ZEROS=false path never reaches the skip branch,
// so it always uses the no-skip instantiation regardless of `use_skip`.
static torch::Tensor geom_prod_fwd_impl(torch::Tensor a, torch::Tensor b,
                                        torch::Tensor packed_sign,
                                        c10::optional<torch::Tensor> packed_valid_opt,
                                        bool use_skip) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32,
                "a must be CUDA float32 contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32,
                "b must be CUDA float32 contiguous");
    TORCH_CHECK(packed_sign.is_cuda() && packed_sign.is_contiguous()
                && packed_sign.scalar_type() == torch::kInt32,
                "packed_sign must be CUDA int32 contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.dim() == 2, "a, b must be (B, dim)");
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    int dim = a.size(1);
    int N = 0;
    while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim, "dim must be a power of two");
    TORCH_CHECK(N >= 5, "N>=5 (dim>=32) required");
    TORCH_CHECK(packed_sign.size(0) == dim && packed_sign.size(1) == dim / 32,
                "packed_sign must be (dim, dim/32)");

    bool has_zeros = packed_valid_opt.has_value();
    const int* valid_ptr = nullptr;
    if (has_zeros) {
        auto& pv = packed_valid_opt.value();
        TORCH_CHECK(pv.is_cuda() && pv.is_contiguous() && pv.scalar_type() == torch::kInt32,
                    "packed_valid must be CUDA int32 contiguous");
        TORCH_CHECK(pv.size(0) == dim && pv.size(1) == dim / 32,
                    "packed_valid must be (dim, dim/32)");
        valid_ptr = pv.data_ptr<int>();
    }

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* a_ptr     = a.data_ptr<float>();
    const float* b_ptr     = b.data_ptr<float>();
    const int*   sign_ptr  = packed_sign.data_ptr<int>();
    float*       c_ptr     = c.data_ptr<float>();

    if (!has_zeros) {
        if      (N == 5)  LAUNCH(5,  4, 2, false, false);
        else if (N == 6)  LAUNCH(6,  4, 2, false, false);
        else if (N == 7)  LAUNCH(7,  4, 2, false, false);
        else if (N == 8)  LAUNCH(8,  4, 2, false, false);
        else if (N == 9)  LAUNCH(9,  16, 4, false, false);
        else if (N == 10) LAUNCH(10, 16, 4, false, false);
        else if (N == 11) LAUNCH(11, 16, 4, false, false);
        else if (N == 12) LAUNCH(12, 16, 4, false, false);
        else if (N == 13) LAUNCH(13, 16, 4, false, false);
        else TORCH_CHECK(false, "unsupported N (5..13)");
    } else if (use_skip) {
        if      (N == 5)  LAUNCH(5,  4, 2, true, true);
        else if (N == 6)  LAUNCH(6,  4, 2, true, true);
        else if (N == 7)  LAUNCH(7,  4, 2, true, true);
        else if (N == 8)  LAUNCH(8,  4, 2, true, true);
        else if (N == 9)  LAUNCH(9,  16, 4, true, true);
        else if (N == 10) LAUNCH(10, 16, 4, true, true);
        else if (N == 11) LAUNCH(11, 16, 4, true, true);
        else if (N == 12) LAUNCH(12, 16, 4, true, true);
        else if (N == 13) LAUNCH(13, 16, 4, true, true);
        else TORCH_CHECK(false, "unsupported N (5..13)");
    } else {
        if      (N == 5)  LAUNCH(5,  4, 2, true, false);
        else if (N == 6)  LAUNCH(6,  4, 2, true, false);
        else if (N == 7)  LAUNCH(7,  4, 2, true, false);
        else if (N == 8)  LAUNCH(8,  4, 2, true, false);
        else if (N == 9)  LAUNCH(9,  16, 4, true, false);
        else if (N == 10) LAUNCH(10, 16, 4, true, false);
        else if (N == 11) LAUNCH(11, 16, 4, true, false);
        else if (N == 12) LAUNCH(12, 16, 4, true, false);
        else if (N == 13) LAUNCH(13, 16, 4, true, false);
        else TORCH_CHECK(false, "unsupported N (5..13)");
    }

    return c;
}

// `geom_prod_fwd` (no chunk-skip) and `geom_prod_fwd_skip` (chunk-skip on the
// HAS_ZEROS path) are exposed as peers so callers/benches can pick whichever
// wins for the LUT density they're using.
torch::Tensor geom_prod_fwd(torch::Tensor a, torch::Tensor b,
                            torch::Tensor packed_sign,
                            c10::optional<torch::Tensor> packed_valid_opt) {
    return geom_prod_fwd_impl(a, b, packed_sign, packed_valid_opt, /*use_skip=*/false);
}

torch::Tensor geom_prod_fwd_skip(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor packed_sign,
                                 c10::optional<torch::Tensor> packed_valid_opt) {
    return geom_prod_fwd_impl(a, b, packed_sign, packed_valid_opt, /*use_skip=*/true);
}


// =============================================================================
// VARIANT: Multi-k per warp. Each warp produces M output blades, holding M
// accumulators. The operand SMEM is read once per chunk and used for all M
// outputs (the per-chunk a-load is amortized M-fold).
// =============================================================================

template<int N, int W, int K, int M, bool HAS_ZEROS, bool USE_SKIP>
__global__ void geom_prod_fwd_multik_kernel(
    const float*  __restrict__ a,
    const float*  __restrict__ b,
    const int*    __restrict__ packed_sign,
    const int*    __restrict__ packed_valid,
    float*        __restrict__ c,
    int batch)
{
    constexpr int DIM = 1 << N;
    constexpr int CHUNKS = DIM >> 5;
    constexpr int N4 = DIM >> 2;
    constexpr int OUTPUTS_PER_BLOCK = W * M;

    int b_idx = blockIdx.x;
    if (b_idx >= batch) return;
    int output_base = blockIdx.y * OUTPUTS_PER_BLOCK;

    extern __shared__ unsigned char smem_raw[];
    float* sA = reinterpret_cast<float*>(smem_raw);
    float* sB = sA + DIM;
    int*   sSig_pool = reinterpret_cast<int*>(sB + DIM);   // M * W * CHUNKS int32s
    int*   sVal_pool = HAS_ZEROS ? (sSig_pool + M * W * CHUNKS) : nullptr;

    int tid = threadIdx.x;
    int blocksize = W * 32;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    // The M output blades this warp produces.
    int ks[M];
    bool ks_valid[M];
    int* warpSig[M];
    int* warpVal[M];
    #pragma unroll
    for (int m = 0; m < M; m++) {
        ks[m] = output_base + m * W + warp_id;
        ks_valid[m] = (ks[m] < DIM);
        warpSig[m] = sSig_pool + (m * W + warp_id) * CHUNKS;
        if constexpr (HAS_ZEROS) {
            warpVal[m] = sVal_pool + (m * W + warp_id) * CHUNKS;
        }
    }

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

    // Stage M sign rows for this warp from global to SMEM.
    #pragma unroll
    for (int m = 0; m < M; m++) {
        if (ks_valid[m]) {
            for (int c_idx = lane; c_idx < CHUNKS; c_idx += 32) {
                warpSig[m][c_idx] = packed_sign[ks[m] * CHUNKS + c_idx];
            }
            if constexpr (HAS_ZEROS) {
                for (int c_idx = lane; c_idx < CHUNKS; c_idx += 32) {
                    warpVal[m][c_idx] = packed_valid[ks[m] * CHUNKS + c_idx];
                }
            }
        }
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float accs[M];
    #pragma unroll
    for (int m = 0; m < M; m++) accs[m] = 0.0f;

    int c_idx = 0;
    #pragma unroll 1
    for (; c_idx + K <= CHUNKS; c_idx += K) {
        #pragma unroll
        for (int u = 0; u < K; u++) {
            int chunk = (c_idx + u) << 5;
            int i = chunk + lane;
            float a_val = sA[i];
            #pragma unroll
            for (int m = 0; m < M; m++) {
                if (!ks_valid[m]) continue;
                int j = i ^ ks[m];
                if constexpr (HAS_ZEROS) {
                    int vword = warpVal[m][c_idx + u];
                    if constexpr (USE_SKIP) {
                        // Warp-uniform chunk-skip when every lane's term is zero.
                        if (vword == 0) continue;
                    }
                    int word = warpSig[m][c_idx + u];
                    float s = 1.0f - 2.0f * float((word >> lane) & 1);
                    float v = float((vword >> lane) & 1);
                    accs[m] += v * s * a_val * sB[j];
                } else {
                    int word = warpSig[m][c_idx + u];
                    float s = 1.0f - 2.0f * float((word >> lane) & 1);
                    accs[m] += s * a_val * sB[j];
                }
            }
        }
    }
    #pragma unroll 1
    for (; c_idx < CHUNKS; c_idx++) {
        int chunk = c_idx << 5;
        int i = chunk + lane;
        float a_val = sA[i];
        #pragma unroll
        for (int m = 0; m < M; m++) {
            if (!ks_valid[m]) continue;
            int j = i ^ ks[m];
            if constexpr (HAS_ZEROS) {
                int vword = warpVal[m][c_idx];
                if constexpr (USE_SKIP) {
                    if (vword == 0) continue;
                }
                int word = warpSig[m][c_idx];
                float s = 1.0f - 2.0f * float((word >> lane) & 1);
                float v = float((vword >> lane) & 1);
                accs[m] += v * s * a_val * sB[j];
            } else {
                int word = warpSig[m][c_idx];
                float s = 1.0f - 2.0f * float((word >> lane) & 1);
                accs[m] += s * a_val * sB[j];
            }
        }
    }

    // Per-output warp reduce + write.
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float acc = accs[m];
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, delta);
        if (lane == 0 && ks_valid[m])
            c[b_idx * DIM + ks[m]] = acc;
    }
}


#define LAUNCH_MULTIK(N_VAL, W_VAL, K_VAL, M_VAL, HAS_Z, USE_S)                                    \
    do {                                                                                            \
        constexpr int DIM = 1 << (N_VAL);                                                           \
        constexpr int CHUNKS = DIM >> 5;                                                            \
        int blocks_y = (DIM + (W_VAL) * (M_VAL) - 1) / ((W_VAL) * (M_VAL));                         \
        dim3 grid(batch, blocks_y);                                                                 \
        int threads = (W_VAL) * 32;                                                                 \
        int sig_smem = (M_VAL) * (W_VAL) * CHUNKS * (int)sizeof(int);                               \
        int smem = (2 * DIM * (int)sizeof(float))                                                   \
                 + sig_smem                                                                         \
                 + ((HAS_Z) ? sig_smem : 0);                                                        \
        if (smem > 48 * 1024)                                                                       \
            TORCH_CHECK(cudaFuncSetAttribute(                                                       \
                geom_prod_fwd_multik_kernel<(N_VAL), (W_VAL), (K_VAL), (M_VAL), (HAS_Z), (USE_S)>,  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,                  \
                "geom_prod multik: dynamic SMEM opt-in failed (need ", smem, " bytes)");            \
        geom_prod_fwd_multik_kernel<(N_VAL), (W_VAL), (K_VAL), (M_VAL), (HAS_Z), (USE_S)>           \
            <<<grid, threads, smem, stream>>>(a_ptr, b_ptr, sign_ptr, valid_ptr,                    \
                                              c_ptr, batch);                                        \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                             \
    } while (0)


// Multi-k (M=2 outputs per warp) GP forward. `use_skip` mirrors
// geom_prod_fwd_impl: warp-uniform chunk-skip on the HAS_ZEROS path only.
static torch::Tensor geom_prod_fwd_multik_impl(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor packed_sign,
                                               c10::optional<torch::Tensor> packed_valid_opt,
                                               bool use_skip) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed_sign.is_cuda() && packed_sign.is_contiguous() && packed_sign.scalar_type() == torch::kInt32);
    TORCH_CHECK(a.sizes() == b.sizes()); TORCH_CHECK(a.dim() == 2);
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0); int dim = a.size(1);
    int N = 0; while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim); TORCH_CHECK(N >= 5);
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
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    const int*   sign_ptr = packed_sign.data_ptr<int>();
    float*       c_ptr = c.data_ptr<float>();

    if (!has_zeros) {
        if      (N == 5)  LAUNCH_MULTIK(5,  4, 2, 2, false, false);
        else if (N == 6)  LAUNCH_MULTIK(6,  4, 2, 2, false, false);
        else if (N == 7)  LAUNCH_MULTIK(7,  4, 2, 2, false, false);
        else if (N == 8)  LAUNCH_MULTIK(8,  4, 2, 2, false, false);
        else if (N == 9)  LAUNCH_MULTIK(9,  16, 4, 2, false, false);
        else if (N == 10) LAUNCH_MULTIK(10, 16, 4, 2, false, false);
        else if (N == 11) LAUNCH_MULTIK(11, 16, 4, 2, false, false);
        else if (N == 12) LAUNCH_MULTIK(12, 16, 4, 2, false, false);
        else if (N == 13) LAUNCH_MULTIK(13, 8,  4, 2, false, false);
        else TORCH_CHECK(false, "unsupported N");
    } else if (use_skip) {
        if      (N == 5)  LAUNCH_MULTIK(5,  4, 2, 2, true, true);
        else if (N == 6)  LAUNCH_MULTIK(6,  4, 2, 2, true, true);
        else if (N == 7)  LAUNCH_MULTIK(7,  4, 2, 2, true, true);
        else if (N == 8)  LAUNCH_MULTIK(8,  4, 2, 2, true, true);
        else if (N == 9)  LAUNCH_MULTIK(9,  16, 4, 2, true, true);
        else if (N == 10) LAUNCH_MULTIK(10, 16, 4, 2, true, true);
        else if (N == 11) LAUNCH_MULTIK(11, 16, 4, 2, true, true);
        else if (N == 12) LAUNCH_MULTIK(12, 16, 4, 2, true, true);
        else if (N == 13) LAUNCH_MULTIK(13, 8,  4, 2, true, true);
        else TORCH_CHECK(false, "unsupported N");
    } else {
        if      (N == 5)  LAUNCH_MULTIK(5,  4, 2, 2, true, false);
        else if (N == 6)  LAUNCH_MULTIK(6,  4, 2, 2, true, false);
        else if (N == 7)  LAUNCH_MULTIK(7,  4, 2, 2, true, false);
        else if (N == 8)  LAUNCH_MULTIK(8,  4, 2, 2, true, false);
        else if (N == 9)  LAUNCH_MULTIK(9,  16, 4, 2, true, false);
        else if (N == 10) LAUNCH_MULTIK(10, 16, 4, 2, true, false);
        else if (N == 11) LAUNCH_MULTIK(11, 16, 4, 2, true, false);
        else if (N == 12) LAUNCH_MULTIK(12, 16, 4, 2, true, false);
        else if (N == 13) LAUNCH_MULTIK(13, 8,  4, 2, true, false);
        else TORCH_CHECK(false, "unsupported N");
    }
    return c;
}

torch::Tensor geom_prod_fwd_multik(torch::Tensor a, torch::Tensor b,
                                   torch::Tensor packed_sign,
                                   c10::optional<torch::Tensor> packed_valid_opt) {
    return geom_prod_fwd_multik_impl(a, b, packed_sign, packed_valid_opt, /*use_skip=*/false);
}

torch::Tensor geom_prod_fwd_multik_skip(torch::Tensor a, torch::Tensor b,
                                        torch::Tensor packed_sign,
                                        c10::optional<torch::Tensor> packed_valid_opt) {
    return geom_prod_fwd_multik_impl(a, b, packed_sign, packed_valid_opt, /*use_skip=*/true);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("geom_prod_fwd", &geom_prod_fwd,
          "Cl(p, q, r) GP forward (cp.async operands, SMEM-staged sign row, K-unroll)",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"),
          pybind11::arg("packed_valid") = c10::optional<torch::Tensor>());
    m.def("geom_prod_fwd_multik", &geom_prod_fwd_multik,
          "Cl(p, q, r) GP forward (multi-k=2 per warp; M outputs share operand load)",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"),
          pybind11::arg("packed_valid") = c10::optional<torch::Tensor>());
    m.def("geom_prod_fwd_skip", &geom_prod_fwd_skip,
          "GP forward with warp-uniform chunk-skip on HAS_ZEROS path "
          "(faster when σ has many fully-zero chunks; slower on denser σ)",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"),
          pybind11::arg("packed_valid") = c10::optional<torch::Tensor>());
    m.def("geom_prod_fwd_multik_skip", &geom_prod_fwd_multik_skip,
          "Multik GP forward with warp-uniform chunk-skip on HAS_ZEROS path",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("packed_sign"),
          pybind11::arg("packed_valid") = c10::optional<torch::Tensor>());
}
