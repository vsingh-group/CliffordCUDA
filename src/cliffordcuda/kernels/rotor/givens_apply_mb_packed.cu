#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


template<int K, int M>
__global__ void givens_mb_packed_fwd_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ cs,
    const int*   __restrict__ packed_ij,
    const int*   __restrict__ packed_sig,
    int num_rot, int ppr, int dim, int batch)
{
    int threads_per_batch = blockDim.x / M;
    int b_local = threadIdx.x / threads_per_batch;
    int tid_b   = threadIdx.x % threads_per_batch;
    int b_global = blockIdx.x * M + b_local;
    if (b_global >= batch) return;

    extern __shared__ float buf[];
    float* my_buf = buf + b_local * dim;

    for (int i = tid_b; i < dim; i += threads_per_batch)
        my_buf[i] = x_in[b_global * dim + i];
    __syncthreads();

    int sig_words_per_rot = (ppr + 31) / 32;

    for (int rot = num_rot - 1; rot >= 0; rot--) {
        float c_w = cs[rot * 2];
        float s_w = cs[rot * 2 + 1];
        int base_ij  = rot * ppr;
        int base_sig = rot * sig_words_per_rot;

        int ii[K], jj[K];
        float sig[K], xi[K], xj[K];

        #pragma unroll
        for (int k = 0; k < K; k++) {
            int p = tid_b + k * threads_per_batch;
            int packed = packed_ij[base_ij + p];
            ii[k] = packed >> 16;
            jj[k] = packed & 0xFFFF;
            int sig_word = packed_sig[base_sig + (p >> 5)];
            int bit = (sig_word >> (p & 31)) & 1;
            sig[k] = bit ? -1.0f : 1.0f;
        }

        #pragma unroll
        for (int k = 0; k < K; k++) {
            xi[k] = my_buf[ii[k]];
            xj[k] = my_buf[jj[k]];
        }

        #pragma unroll
        for (int k = 0; k < K; k++) {
            float ssig = s_w * sig[k];
            my_buf[ii[k]] =  c_w * xi[k] + ssig * xj[k];
            my_buf[jj[k]] = -ssig * xi[k] + c_w * xj[k];
        }
        __syncthreads();
    }

    for (int i = tid_b; i < dim; i += threads_per_batch)
        x_out[b_global * dim + i] = my_buf[i];
}


template<int K, int M>
__global__ void givens_mb_packed_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x_out,
    float* __restrict__ grad_cs_all,
    const float* __restrict__ cs,
    const int*   __restrict__ packed_ij,
    const int*   __restrict__ packed_sig,
    int num_rot, int ppr, int dim, int batch)
{
    int threads_per_batch = blockDim.x / M;
    int b_local = threadIdx.x / threads_per_batch;
    int tid_b   = threadIdx.x % threads_per_batch;
    int b_global = blockIdx.x * M + b_local;
    if (b_global >= batch) return;

    int warps_per_batch = threads_per_batch / 32;
    int warp_in_batch = tid_b / 32;
    int lane = tid_b % 32;

    extern __shared__ float smem[];
    float* grad_buf = smem;
    float* val_buf  = smem + M * dim;
    float* warp_buf = smem + 2 * M * dim;

    float* my_grad = grad_buf + b_local * dim;
    float* my_val  = val_buf  + b_local * dim;
    float* my_wbuf = warp_buf + b_local * warps_per_batch * 2;

    for (int i = tid_b; i < dim; i += threads_per_batch) {
        my_grad[i] = grad_out[b_global * dim + i];
        my_val[i]  = x_out[b_global * dim + i];
    }
    __syncthreads();

    int sig_words_per_rot = (ppr + 31) / 32;

    for (int rot = 0; rot < num_rot; rot++) {
        float c_w = cs[rot * 2];
        float s_w = cs[rot * 2 + 1];
        int base_ij  = rot * ppr;
        int base_sig = rot * sig_words_per_rot;

        float local_grad_c = 0.0f;
        float local_grad_s = 0.0f;

        int ii[K], jj[K];
        float sig[K], vi[K], vj[K], gi[K], gj[K];

        #pragma unroll
        for (int k = 0; k < K; k++) {
            int p = tid_b + k * threads_per_batch;
            int packed = packed_ij[base_ij + p];
            ii[k] = packed >> 16;
            jj[k] = packed & 0xFFFF;
            int sig_word = packed_sig[base_sig + (p >> 5)];
            int bit = (sig_word >> (p & 31)) & 1;
            sig[k] = bit ? -1.0f : 1.0f;
        }

        #pragma unroll
        for (int k = 0; k < K; k++) {
            vi[k] = my_val[ii[k]];
            vj[k] = my_val[jj[k]];
            gi[k] = my_grad[ii[k]];
            gj[k] = my_grad[jj[k]];
        }

        #pragma unroll
        for (int k = 0; k < K; k++) {
            float ssig = s_w * sig[k];
            float vi_pre = c_w*vi[k] - ssig*vj[k];
            float vj_pre = ssig*vi[k] + c_w*vj[k];
            my_val[ii[k]] = vi_pre;
            my_val[jj[k]] = vj_pre;

            local_grad_c += gi[k]*vi_pre + gj[k]*vj_pre;
            local_grad_s += gi[k]*sig[k]*vj_pre - gj[k]*sig[k]*vi_pre;

            my_grad[ii[k]] = c_w*gi[k] - ssig*gj[k];
            my_grad[jj[k]] = ssig*gi[k] + c_w*gj[k];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            local_grad_c += __shfl_down_sync(0xffffffff, local_grad_c, offset);
            local_grad_s += __shfl_down_sync(0xffffffff, local_grad_s, offset);
        }
        if (lane == 0) {
            my_wbuf[warp_in_batch * 2]     = local_grad_c;
            my_wbuf[warp_in_batch * 2 + 1] = local_grad_s;
        }
        __syncthreads();

        if (tid_b == 0) {
            float sc = 0, ss = 0;
            for (int w = 0; w < warps_per_batch; w++) {
                sc += my_wbuf[w * 2];
                ss += my_wbuf[w * 2 + 1];
            }
            grad_cs_all[b_global * num_rot * 2 + rot * 2]     = sc;
            grad_cs_all[b_global * num_rot * 2 + rot * 2 + 1] = ss;
        }
        __syncthreads();
    }
}


#define DISPATCH_KM(KMAC, MMAC) \
    do { \
        if (K == 1  && M == MMAC) { KMAC(1,  MMAC); return; } \
        if (K == 2  && M == MMAC) { KMAC(2,  MMAC); return; } \
        if (K == 4  && M == MMAC) { KMAC(4,  MMAC); return; } \
        if (K == 8  && M == MMAC) { KMAC(8,  MMAC); return; } \
        if (K == 16 && M == MMAC) { KMAC(16, MMAC); return; } \
    } while(0)


static void launch_fwd(
    int K, int M, int batch, int dim, int num_rot, int ppr,
    const float* x_in, float* x_out, const float* cs,
    const int* packed_ij, const int* packed_sig)
{
    int threads_per_batch = ppr / K;
    int threads = M * threads_per_batch;
    int smem = M * dim * (int)sizeof(float);
    int grid = (batch + M - 1) / M;
    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH(KV, MV) \
        if (smem > 48*1024) \
            TORCH_CHECK(cudaFuncSetAttribute(givens_mb_packed_fwd_kernel<KV, MV>, \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "dynamic SMEM opt-in failed"); \
        givens_mb_packed_fwd_kernel<KV, MV><<<grid, threads, smem, stream>>>( \
            x_in, x_out, cs, packed_ij, packed_sig, num_rot, ppr, dim, batch); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();

    DISPATCH_KM(LAUNCH, 1);
    DISPATCH_KM(LAUNCH, 2);
    DISPATCH_KM(LAUNCH, 4);
    DISPATCH_KM(LAUNCH, 8);
    TORCH_CHECK(false, "unsupported (K, M) combination");
    #undef LAUNCH
}


static void launch_bwd(
    int K, int M, int batch, int dim, int num_rot, int ppr,
    const float* grad_out, const float* x_out, float* grad_cs_all,
    const float* cs, const int* packed_ij, const int* packed_sig)
{
    int threads_per_batch = ppr / K;
    int threads = M * threads_per_batch;
    int warps_per_batch = threads_per_batch / 32;
    int smem = (2 * M * dim + M * warps_per_batch * 2) * (int)sizeof(float);
    int grid = (batch + M - 1) / M;
    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH(KV, MV) \
        if (smem > 48*1024) \
            TORCH_CHECK(cudaFuncSetAttribute(givens_mb_packed_bwd_kernel<KV, MV>, \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "dynamic SMEM opt-in failed"); \
        givens_mb_packed_bwd_kernel<KV, MV><<<grid, threads, smem, stream>>>( \
            grad_out, x_out, grad_cs_all, cs, packed_ij, packed_sig, \
            num_rot, ppr, dim, batch); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();

    DISPATCH_KM(LAUNCH, 1);
    DISPATCH_KM(LAUNCH, 2);
    DISPATCH_KM(LAUNCH, 4);
    DISPATCH_KM(LAUNCH, 8);
    TORCH_CHECK(false, "unsupported (K, M) combination");
    #undef LAUNCH
}


torch::Tensor givens_mb_packed_fwd(
    torch::Tensor x, torch::Tensor cs,
    torch::Tensor packed_ij, torch::Tensor packed_sig,
    int K, int M)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    const at::cuda::CUDAGuard guard(x.device());
    int batch = x.size(0), dim = x.size(1);
    int num_rot = cs.size(0), ppr = packed_ij.size(1);
    TORCH_CHECK(ppr % K == 0, "ppr must be divisible by K");
    int threads_per_batch = ppr / K;
    TORCH_CHECK(threads_per_batch % 32 == 0, "threads_per_batch must be multiple of 32");

    auto y = torch::empty_like(x);
    launch_fwd(K, M, batch, dim, num_rot, ppr,
               x.data_ptr<float>(), y.data_ptr<float>(), cs.data_ptr<float>(),
               packed_ij.data_ptr<int>(), packed_sig.data_ptr<int>());
    return y;
}


torch::Tensor givens_mb_packed_bwd(
    torch::Tensor grad_out, torch::Tensor x_out, torch::Tensor cs,
    torch::Tensor packed_ij, torch::Tensor packed_sig,
    int K, int M)
{
    TORCH_CHECK(grad_out.is_cuda() && grad_out.is_contiguous());
    const at::cuda::CUDAGuard guard(grad_out.device());
    int batch = grad_out.size(0), dim = grad_out.size(1);
    int num_rot = cs.size(0), ppr = packed_ij.size(1);
    TORCH_CHECK(ppr % K == 0, "ppr must be divisible by K");
    int threads_per_batch = ppr / K;
    TORCH_CHECK(threads_per_batch % 32 == 0, "threads_per_batch must be multiple of 32");

    auto grad_cs_all = torch::empty({batch, num_rot, 2}, cs.options());
    launch_bwd(K, M, batch, dim, num_rot, ppr,
               grad_out.data_ptr<float>(), x_out.data_ptr<float>(),
               grad_cs_all.data_ptr<float>(), cs.data_ptr<float>(),
               packed_ij.data_ptr<int>(), packed_sig.data_ptr<int>());
    return grad_cs_all.sum(0);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("givens_mb_packed_fwd", &givens_mb_packed_fwd,
          "MB Givens forward with packed indices (no perm)");
    m.def("givens_mb_packed_bwd", &givens_mb_packed_bwd,
          "MB Givens backward with packed indices (no perm)");
}
