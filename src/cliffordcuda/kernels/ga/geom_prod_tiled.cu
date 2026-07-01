// Tiled Cl(p, q, r) geometric product for n = L + H, where 2^n exceeds the
// per-block shared memory the standard kernel needs (so n is past that kernel's
// device-dependent limit).
//
// Split the bit-pattern index into high H bits (tile p) and low L bits (i').
// Output index = i ^ j, so tile-pair (p, q) contributes entirely to output tile
// r = p ^ q. The sign factorizes:
//
//   sigma(i, j) = sigma_high(p, q)
//               * (-1)^(popcount(p) * popcount(j_low))      // cross term
//               * sigma_low(i_low, j_low)
//
// Each warp computes ONE low output blade k' of ONE output tile r for ONE batch
// element, looping the X tile-pairs (p, p^r): the two tiles are staged via
// cp.async (float4), the cross term is folded (grade involution) into a tile, and
// a K-unrolled loop accumulates with the sigma_low sign applied by sign-bit XOR.
// cross_mode selects the forward sum (0) or one of the two backward sums (1, 2),
// so forward and backward both run on this one kernel. L (and H) are RUNTIME
// arguments, so the caller picks the largest L that fits the actual device.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>      // cp.async (Ampere+)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


__global__ void geom_prod_tiled_kernel(
    const float*       __restrict__ a,
    const float*       __restrict__ b,
    const int*         __restrict__ sign_low,
    const int*         __restrict__ valid_low,
    const signed char* __restrict__ sigma_high,
    float*             __restrict__ c,
    int batch, int L, int H, int W, int has_zeros, int cross_mode)
{
    constexpr int K = 4;
    int M = 1 << L;
    int X = 1 << H;
    long FULL = (long)X * M;
    int CHUNKS = M >> 5;
    int M4 = M >> 2;

    int b_idx = blockIdx.x;
    if (b_idx >= batch) return;
    int r        = blockIdx.y;
    int out_base = blockIdx.z * W;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int kp = out_base + warp_id;
    int blocksize = W * 32;

    extern __shared__ unsigned char smem_raw[];
    float* sA   = reinterpret_cast<float*>(smem_raw);
    float* sB   = sA + M;
    int*   sSig = reinterpret_cast<int*>(sB + M);
    int*   sVal = has_zeros ? (sSig + W * CHUNKS) : nullptr;
    float4* sA4 = reinterpret_cast<float4*>(sA);
    float4* sB4 = reinterpret_cast<float4*>(sB);

    if (kp < M) {
        int* row = sSig + warp_id * CHUNKS;
        for (int ci = lane; ci < CHUNKS; ci += 32)
            row[ci] = sign_low[kp * CHUNKS + ci];
        if (has_zeros) {
            int* vr = sVal + warp_id * CHUNKS;
            for (int ci = lane; ci < CHUNKS; ci += 32)
                vr[ci] = valid_low[kp * CHUNKS + ci];
        }
    }
    __syncthreads();

    int* sSig_w = sSig + warp_id * CHUNKS;
    int* sVal_w = has_zeros ? (sVal + warp_id * CHUNKS) : nullptr;
    float acc = 0.0f;
    const float* a_base = a + (long)b_idx * FULL;
    const float* b_base = b + (long)b_idx * FULL;

    for (int p = 0; p < X; p++) {
        int q = p ^ r;
        int sh = (int)sigma_high[p * X + q];
        if (sh == 0) continue;

        __syncthreads();
        const float4* a4 = reinterpret_cast<const float4*>(a_base + (long)p * M);
        const float4* b4 = reinterpret_cast<const float4*>(b_base + (long)q * M);
        for (int i = tid; i < M4; i += blocksize) {
            __pipeline_memcpy_async(&sA4[i], &a4[i], 16);
            __pipeline_memcpy_async(&sB4[i], &b4[i], 16);
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // Cross term, grade involution by mode (see geom_prod_tiled.py backward):
        //   0 fwd:   fold sB (b) by popcount(p)   1 bwd_a: fold sA (a) by popcount(r)
        //   2 bwd_b: no operand fold (scalar folded into sh per warp below)
        bool fold_sB = (cross_mode == 0) && (__popc(p) & 1);
        bool fold_sA = (cross_mode == 1) && (__popc(r) & 1);
        if (fold_sB) {
            for (int i = tid; i < M; i += blocksize)
                if (__popc(i) & 1) sB[i] = -sB[i];
            __syncthreads();
        } else if (fold_sA) {
            for (int i = tid; i < M; i += blocksize)
                if (__popc(i) & 1) sA[i] = -sA[i];
            __syncthreads();
        }

        if (kp < M) {
            float partial = 0.0f;
            int ci = 0;
            for (; ci + K <= CHUNKS; ci += K) {
                float local = 0.0f;
                #pragma unroll
                for (int u = 0; u < K; u++) {
                    int i = ((ci + u) << 5) + lane;
                    int j = i ^ kp;
                    int word = sSig_w[ci + u];
                    if (has_zeros) {
                        int vword = sVal_w[ci + u];
                        if (vword == 0) continue;         // warp-uniform skip
                        float pr = sA[i] * sB[j];
                        pr = __uint_as_float(__float_as_uint(pr) ^ ((unsigned)((word >> lane) & 1) << 31));
                        local += float((vword >> lane) & 1) * pr;
                    } else {
                        float pr = sA[i] * sB[j];
                        pr = __uint_as_float(__float_as_uint(pr) ^ ((unsigned)((word >> lane) & 1) << 31));
                        local += pr;
                    }
                }
                partial += local;
            }
            for (; ci < CHUNKS; ci++) {
                int i = (ci << 5) + lane;
                int j = i ^ kp;
                int word = sSig_w[ci];
                if (has_zeros) {
                    int vword = sVal_w[ci];
                    if (vword == 0) continue;
                    float pr = sA[i] * sB[j];
                    pr = __uint_as_float(__float_as_uint(pr) ^ ((unsigned)((word >> lane) & 1) << 31));
                    partial += float((vword >> lane) & 1) * pr;
                } else {
                    float pr = sA[i] * sB[j];
                    pr = __uint_as_float(__float_as_uint(pr) ^ ((unsigned)((word >> lane) & 1) << 31));
                    partial += pr;
                }
            }
            float shf = float(sh);                        // bwd_b cross is a per-warp scalar
            if (cross_mode == 2 && (__popc(p) & 1) && (__popc(kp) & 1)) shf = -shf;
            acc += shf * partial;
        }
    }

    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, d);
    if (lane == 0 && kp < M)
        c[(long)b_idx * FULL + (long)r * M + kp] = acc;
}


// Max dynamic shared memory per block the device allows a kernel to opt into.
// This is the real budget that bounds the tile dimension L; torch's
// get_device_properties does not expose it here, so query CUDA directly.
int64_t max_dynamic_smem_optin(int device) {
    int v = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(
                    &v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess,
                "cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin) failed");
    return v;
}


torch::Tensor geom_prod_tiled_fwd(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor sign_low,
                                  c10::optional<torch::Tensor> valid_low_opt,
                                  torch::Tensor sigma_high,
                                  int w_override, int cross_mode) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32,
                "a must be CUDA float32 contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32,
                "b must be CUDA float32 contiguous");
    TORCH_CHECK(a.sizes() == b.sizes() && a.dim() == 2, "a, b must be (B, dim) and equal");
    TORCH_CHECK(sign_low.is_cuda() && sign_low.is_contiguous()
                && sign_low.scalar_type() == torch::kInt32, "sign_low CUDA int32 contiguous");
    TORCH_CHECK(sigma_high.is_cuda() && sigma_high.is_contiguous()
                && sigma_high.scalar_type() == torch::kInt8, "sigma_high CUDA int8 contiguous");
    TORCH_CHECK(cross_mode >= 0 && cross_mode <= 2, "cross_mode must be 0 (fwd), 1 or 2 (bwd)");
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    long dim = a.size(1);
    int M = sign_low.size(0);
    int L = 0; while ((1 << L) < M) L++;
    TORCH_CHECK((1 << L) == M && sign_low.size(1) == M / 32, "sign_low must be (2^L, 2^L/32)");
    int X = sigma_high.size(0);
    int H = 0; while ((1 << H) < X) H++;
    TORCH_CHECK((1 << H) == X && sigma_high.size(1) == X, "sigma_high must be (2^H, 2^H)");
    TORCH_CHECK(dim == (long)X * M, "dim must equal X*M = 2^(L+H)");
    TORCH_CHECK(L >= 5, "L must be >= 5 (dim of inner product >= 32)");

    bool has_zeros = valid_low_opt.has_value();
    const int* valid_ptr = nullptr;
    if (has_zeros) {
        auto& v = valid_low_opt.value();
        TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kInt32
                    && v.size(0) == M && v.size(1) == M / 32, "valid_low must be (2^L, 2^L/32) int32");
        valid_ptr = v.data_ptr<int>();
    }

    int W = (w_override > 0) ? w_override : ((L >= 9) ? 16 : 4);   // warps/block
    TORCH_CHECK(W >= 1 && W * 32 <= 1024, "W must be in [1, 32] (W*32 threads <= 1024)");
    int CHUNKS = M >> 5;
    dim3 grid(batch, X, (M + W - 1) / W);
    int threads = W * 32;
    int sig_smem = W * CHUNKS * (int)sizeof(int);
    int smem = 2 * M * (int)sizeof(float) + sig_smem + (has_zeros ? sig_smem : 0);
    if (smem > 48 * 1024)
        TORCH_CHECK(cudaFuncSetAttribute((const void*)geom_prod_tiled_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                    "geom_prod_tiled: dynamic SMEM opt-in failed (need ", smem, " bytes; "
                    "device cannot fit L=", L, " -- caller should pick a smaller L)");

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    geom_prod_tiled_kernel<<<grid, threads, smem, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), sign_low.data_ptr<int>(),
        valid_ptr, sigma_high.data_ptr<signed char>(), c.data_ptr<float>(),
        batch, L, H, W, has_zeros ? 1 : 0, cross_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("geom_prod_tiled_fwd", &geom_prod_tiled_fwd,
          "Tiled Cl(p,q,r) geometric product (cp.async + K-unroll + sign-bit XOR); "
          "cross_mode 0=forward, 1/2=backward.",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("sign_low"),
          pybind11::arg("valid_low") = c10::optional<torch::Tensor>(),
          pybind11::arg("sigma_high"), pybind11::arg("w_override") = 0,
          pybind11::arg("cross_mode") = 0);
    m.def("max_dynamic_smem_optin", &max_dynamic_smem_optin,
          "Max dynamic shared memory (bytes) a block can opt into on the device",
          pybind11::arg("device"));
}
