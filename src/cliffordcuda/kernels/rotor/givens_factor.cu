#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void givens_factor_fwd_kernel(
    const float* __restrict__ R_in,
    float*       __restrict__ cs_out,
    float*       __restrict__ A_out,
    int n, int n_rot)
{
    int b = blockIdx.x;

    extern __shared__ float smem[];
    float* A      = smem;
    float* cs_buf = smem + n * n; 

    for (int i = threadIdx.x; i < n * n; i += blockDim.x)
        A[i] = R_in[b * n * n + i];
    __syncthreads();

    int rot_count = 0;
    for (int p = 0; p < n - 1; p++) {
        for (int q = p + 1; q < n; q++) {
            if (threadIdx.x == 0) {
                float a     = A[p * n + p];
                float bv    = A[q * n + p];
                float r_val = sqrtf(a * a + bv * bv);
                float inv_r = 1.0f / fmaxf(r_val, 1e-12f);
                float c     = a  * inv_r;
                float s_raw = -bv * inv_r;
                cs_buf[0] = c;
                cs_buf[1] = s_raw;

                int out_idx = n_rot - 1 - rot_count;
                cs_out[b * n_rot * 2 + out_idx * 2 + 0] = c;
                cs_out[b * n_rot * 2 + out_idx * 2 + 1] = -s_raw;
            }
            __syncthreads();

            float c     = cs_buf[0];
            float s_raw = cs_buf[1];

            for (int k = threadIdx.x; k < n; k += blockDim.x) {
                float Apk = A[p * n + k];
                float Aqk = A[q * n + k];
                A[p * n + k] =  c * Apk - s_raw * Aqk;
                A[q * n + k] =  s_raw * Apk + c * Aqk;
            }
            __syncthreads();

            rot_count++;
        }
    }

    for (int i = threadIdx.x; i < n * n; i += blockDim.x)
        A_out[b * n * n + i] = A[i];
}

__global__ void givens_factor_bwd_kernel(
    const float* __restrict__ cs_in,
    const float* __restrict__ A_final,
    const float* __restrict__ grad_cs_in,
    float*       __restrict__ grad_R_out,
    int n, int n_rot)
{
    int b = blockIdx.x;

    extern __shared__ float smem[];
    float* A       = smem;
    float* grad_A  = smem + n * n;
    float* cs_buf  = smem + 2 * n * n;
    float* gc_buf  = smem + 2 * n * n + 2;

    for (int i = threadIdx.x; i < n * n; i += blockDim.x)
        A[i] = A_final[b * n * n + i];

    for (int i = threadIdx.x; i < n * n; i += blockDim.x)
        grad_A[i] = 0.0f;
    __syncthreads();

    for (int rot_rev = 0; rot_rev < n_rot; rot_rev++) {
        int cs_idx = rot_rev;
        int fwd_idx = n_rot - 1 - rot_rev;
        int pp = 0, qq = 0;
        {
            int cnt = 0;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (cnt == fwd_idx) { pp = p; qq = q; }
                    cnt++;
                }
            }
        }

        if (threadIdx.x == 0) {
            cs_buf[0] = cs_in[b * n_rot * 2 + cs_idx * 2 + 0];
            cs_buf[1] = -(cs_in[b * n_rot * 2 + cs_idx * 2 + 1]);
        }
        __syncthreads();

        float c     = cs_buf[0];
        float s_raw = cs_buf[1];

        for (int k = threadIdx.x; k < n; k += blockDim.x) {
            float Apk_post = A[pp * n + k];
            float Aqk_post = A[qq * n + k];
            A[pp * n + k] =  c * Apk_post + s_raw * Aqk_post;
            A[qq * n + k] = -s_raw * Apk_post + c * Aqk_post;
        }
        __syncthreads();

        float grad_c_up, grad_s_raw_up;
        if (threadIdx.x == 0) {
            grad_c_up     = grad_cs_in[b * n_rot * 2 + cs_idx * 2 + 0];
            grad_s_raw_up = -grad_cs_in[b * n_rot * 2 + cs_idx * 2 + 1];
            gc_buf[0] = grad_c_up;
            gc_buf[1] = grad_s_raw_up;
        }
        __syncthreads();
        grad_c_up     = gc_buf[0];
        grad_s_raw_up = gc_buf[1];

        float local_gc = 0.0f;
        float local_gs = 0.0f;

        for (int k = threadIdx.x; k < n; k += blockDim.x) {
            float gAp = grad_A[pp * n + k];
            float gAq = grad_A[qq * n + k];
            float Apk = A[pp * n + k];
            float Aqk = A[qq * n + k];

            local_gc += gAp * Apk + gAq * Aqk;
            local_gs += -gAp * Aqk + gAq * Apk;

            grad_A[pp * n + k] =  c * gAp + s_raw * gAq;
            grad_A[qq * n + k] = -s_raw * gAp + c * gAq;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            gc_buf[0] = 0.0f;
            gc_buf[1] = 0.0f;
        }
        __syncthreads();
        atomicAdd(&gc_buf[0], local_gc);
        atomicAdd(&gc_buf[1], local_gs);
        __syncthreads();

        float total_gc = gc_buf[0] + grad_c_up;
        float total_gs = gc_buf[1] + grad_s_raw_up;

        if (threadIdx.x == 0) {
            float a = A[pp * n + pp];
            float bv = A[qq * n + pp];
            float r_sq = a * a + bv * bv;
            float r_val = sqrtf(r_sq);
            float inv_r3 = 1.0f / fmaxf(r_val * r_sq, 1e-24f);

            float dc_da =  bv * bv * inv_r3;
            float dc_db = -a  * bv * inv_r3;
            float ds_da =  a  * bv * inv_r3;
            float ds_db = -a  * a  * inv_r3;

            grad_A[pp * n + pp] += total_gc * dc_da + total_gs * ds_da;
            grad_A[qq * n + pp] += total_gc * dc_db + total_gs * ds_db;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < n * n; i += blockDim.x)
        grad_R_out[b * n * n + i] = grad_A[i];
}

std::vector<torch::Tensor> givens_factor_fwd_v2(torch::Tensor R, int n_rot)
{
    TORCH_CHECK(R.is_cuda() && R.is_contiguous());
    const at::cuda::CUDAGuard device_guard(R.device());

    int B = R.size(0);
    int n = R.size(1);

    auto cs_out = torch::empty({B, n_rot, 2}, R.options());
    auto A_out  = torch::empty({B, n, n}, R.options());

    int threads = n;
    int smem = (n * n + 2) * sizeof(float);

    givens_factor_fwd_kernel<<<B, threads, smem,
                               at::cuda::getCurrentCUDAStream()>>>(
        R.data_ptr<float>(),
        cs_out.data_ptr<float>(),
        A_out.data_ptr<float>(),
        n, n_rot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {cs_out, A_out};
}


torch::Tensor givens_factor_bwd_v2(
    torch::Tensor cs,
    torch::Tensor A_final,
    torch::Tensor grad_cs,
    int n_rot)
{
    TORCH_CHECK(cs.is_cuda() && cs.is_contiguous());
    TORCH_CHECK(A_final.is_cuda() && A_final.is_contiguous());
    TORCH_CHECK(grad_cs.is_cuda() && grad_cs.is_contiguous());
    const at::cuda::CUDAGuard device_guard(cs.device());

    int B = cs.size(0);
    int n = A_final.size(1);

    auto grad_R = torch::empty({B, n, n}, cs.options());

    int threads = n;
    int smem = (2 * n * n + 4) * sizeof(float);

    givens_factor_bwd_kernel<<<B, threads, smem,
                               at::cuda::getCurrentCUDAStream()>>>(
        cs.data_ptr<float>(),
        A_final.data_ptr<float>(),
        grad_cs.data_ptr<float>(),
        grad_R.data_ptr<float>(),
        n, n_rot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_R;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("givens_factor_fwd_v2", &givens_factor_fwd_v2,
          "Batched Givens factorisation (forward, saves state for backward)");
    m.def("givens_factor_bwd_v2", &givens_factor_bwd_v2,
          "Batched Givens factorisation (backward)");
}
