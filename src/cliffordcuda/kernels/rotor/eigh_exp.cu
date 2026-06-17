#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>

__global__ void closed_form_R_fwd_kernel(
    const float* __restrict__ B,
    const float* __restrict__ eigenvalues,
    const float* __restrict__ Q,
    float*       __restrict__ R_out,
    float*       __restrict__ V_out,
    float*       __restrict__ phi_out,
    const int* __restrict__ pick_idx,
    int N, int K)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int NN = N * N;

    extern __shared__ float smem[];
    float* B_s   = smem;
    float* R_s   = smem + NN;
    float* V_s   = smem + 2*NN;
    float* U_s   = smem + 2*NN + K*N;
    float* phi_s = smem + 2*NN + 2*K*N;

    for (int i = tid; i < NN; i += blockDim.x)
        B_s[i] = B[b * NN + i];
    __syncthreads();

    for (int k = tid; k < K; k += blockDim.x) {
        int idx = pick_idx[k];
        phi_s[k] = sqrtf(fmaxf(eigenvalues[b * N + idx], 1e-12f));
    }
    for (int k = 0; k < K; k++) {
        int idx = pick_idx[k];
        for (int i = tid; i < N; i += blockDim.x)
            V_s[k * N + i] = Q[b * NN + i * N + idx];
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        float inv_phi = 1.0f / fmaxf(phi_s[k], 1e-12f);
        for (int i = tid; i < N; i += blockDim.x) {
            float s = 0.0f;
            for (int j = 0; j < N; j++)
                s += B_s[i * N + j] * V_s[k * N + j];
            U_s[k * N + i] = s * inv_phi;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < NN; idx += blockDim.x) {
        int r = idx / N, c = idx % N;
        float val = (r == c) ? 1.0f : 0.0f;
        for (int k = 0; k < K; k++) {
            float p = phi_s[k];
            float cos_m1 = cosf(p) - 1.0f;
            float sin_p = sinf(p);
            float vr = V_s[k*N+r], vc = V_s[k*N+c];
            float ur = U_s[k*N+r], uc = U_s[k*N+c];
            val += cos_m1 * (vr*vc + ur*uc) + sin_p * (ur*vc - vr*uc);
        }
        R_s[idx] = val;
    }
    __syncthreads();

    for (int i = tid; i < NN; i += blockDim.x)
        R_out[b * NN + i] = R_s[i];
    for (int k = 0; k < K; k++)
        for (int i = tid; i < N; i += blockDim.x)
            V_out[b * K * N + k * N + i] = V_s[k * N + i];
    for (int k = tid; k < K; k += blockDim.x)
        phi_out[b * K + k] = phi_s[k];
}

__global__ void eigh_exp_bwd_kernel(
    const float* __restrict__ grad_R_in,
    const float* __restrict__ B_in,
    const float* __restrict__ V_in,
    const float* __restrict__ phi_in,
    const float* __restrict__ eig_clean,
    const float* __restrict__ Q_in,
    const int*   __restrict__ skew_i,
    const int*   __restrict__ skew_j,
    float*       __restrict__ grad_biv_out,
    int N, int K, int n_biv)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int NN = N * N;

    extern __shared__ float smem[];
    float* B_s    = smem;
    float* gR_s   = smem + NN;
    float* V_s    = smem + 2*NN;
    float* U_s    = smem + 2*NN + K*N;
    float* phi_s  = smem + 2*NN + 2*K*N;
    float* gV_s   = smem + 2*NN + 2*K*N + K;
    float* gU_s   = smem + 2*NN + 3*K*N + K;
    float* gB_s   = smem + 2*NN + 4*K*N + K;
    float* gM_s   = smem + 3*NN + 4*K*N + K;
    float* Q_s    = smem + 4*NN + 4*K*N + K;
    float* eig_s  = smem + 5*NN + 4*K*N + K;
    float* gph_s  = smem + 5*NN + 4*K*N + K + N;
    float* tmp_s  = smem + 5*NN + 4*K*N + 2*K + N;

    for (int i = tid; i < NN; i += blockDim.x) {
        B_s[i]  = B_in[b * NN + i];
        gR_s[i] = grad_R_in[b * NN + i];
        Q_s[i]  = Q_in[b * NN + i];
        gB_s[i] = 0.0f;
        gM_s[i] = 0.0f;
    }
    for (int i = tid; i < K*N; i += blockDim.x) {
        V_s[i] = V_in[b * K * N + i];
        gV_s[i] = 0.0f;
        gU_s[i] = 0.0f;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        phi_s[i] = phi_in[b * K + i];
        gph_s[i] = 0.0f;
    }
    for (int i = tid; i < N; i += blockDim.x)
        eig_s[i] = eig_clean[b * N + i];
    __syncthreads();

    for (int k = 0; k < K; k++) {
        float inv_phi = 1.0f / fmaxf(phi_s[k], 1e-12f);
        for (int i = tid; i < N; i += blockDim.x) {
            float s = 0.0f;
            for (int j = 0; j < N; j++)
                s += B_s[i*N+j] * V_s[k*N+j];
            U_s[k*N+i] = s * inv_phi;
        }
        __syncthreads();
    }

    for (int k = 0; k < K; k++) {
        float p = phi_s[k];
        float cos_m1 = cosf(p) - 1.0f;
        float sin_p = sinf(p);
        float neg_sin = -sin_p;
        float cos_p = cosf(p);

        float local_gph = 0.0f;

        for (int i = tid; i < N; i += blockDim.x) {
            float gv_i = 0.0f, gu_i = 0.0f;
            for (int j = 0; j < N; j++) {
                float gR_ij = gR_s[i*N+j];
                float gR_ji = gR_s[j*N+i];
                float sym = gR_ij + gR_ji;
                float asym = gR_ij - gR_ji;
                float vj = V_s[k*N+j];
                float uj = U_s[k*N+j];
                gv_i += cos_m1 * sym * vj - sin_p * asym * uj;
                gu_i += cos_m1 * sym * uj + sin_p * asym * vj;
            }
            gV_s[k*N+i] = gv_i;
            gU_s[k*N+i] = gu_i;

            float vi = V_s[k*N+i], ui = U_s[k*N+i];
            for (int j = 0; j < N; j++) {
                float gR_ij = gR_s[i*N+j];
                float vj = V_s[k*N+j], uj = U_s[k*N+j];
                local_gph += gR_ij * (neg_sin*(vi*vj + ui*uj) + cos_p*(ui*vj - vi*uj));
            }
        }
        __syncthreads();

        tmp_s[tid] = local_gph;
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int t = 0; t < blockDim.x; t++) sum += tmp_s[t];
            gph_s[k] = sum;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < NN; idx += blockDim.x) {
        int r = idx / N, c = idx % N;
        float s = 0.0f;
        for (int k = 0; k < K; k++)
            s += gU_s[k*N+r] * V_s[k*N+c] / fmaxf(phi_s[k], 1e-12f);
        gB_s[idx] = s;
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        float inv_phi = 1.0f / fmaxf(phi_s[k], 1e-12f);
        for (int i = tid; i < N; i += blockDim.x) {
            float s = 0.0f;
            for (int j = 0; j < N; j++)
                s += B_s[j*N+i] * gU_s[k*N+j];
            gV_s[k*N+i] += s * inv_phi;
        }
        __syncthreads();
    }

    for (int k = 0; k < K; k++) {
        float inv_phi = 1.0f / fmaxf(phi_s[k], 1e-12f);
        float local = 0.0f;
        for (int i = tid; i < N; i += blockDim.x)
            local += -gU_s[k*N+i] * U_s[k*N+i] * inv_phi;
        tmp_s[tid] = local;
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int t = 0; t < blockDim.x; t++) sum += tmp_s[t];
            gph_s[k] += sum;
        }
        __syncthreads();
    }

    for (int k = tid; k < K; k += blockDim.x)
        gph_s[k] = gph_s[k] / (2.0f * fmaxf(phi_s[k], 1e-12f));
    __syncthreads();

    for (int k = 0; k < K; k++) {
        float lam_k = phi_s[k] * phi_s[k];

        float gps_k = gph_s[k];
        for (int idx = tid; idx < NN; idx += blockDim.x) {
            int r = idx / N, c = idx % N;
            gM_s[idx] += gps_k * V_s[k*N+r] * V_s[k*N+c];
        }
        __syncthreads();

        for (int i = tid; i < N; i += blockDim.x) {
            float z_i = 0.0f;
            for (int j = 0; j < N; j++) {
                float diff = eig_s[j] - lam_k;
                if (fabsf(diff) > 1e-4f) {
                    float coeff_j = 0.0f;
                    for (int l = 0; l < N; l++)
                        coeff_j += Q_s[l*N+j] * gV_s[k*N+l];
                    z_i += Q_s[i*N+j] * coeff_j / diff;
                }
            }
            tmp_s[i] = z_i;
        }
        __syncthreads();

        for (int idx = tid; idx < NN; idx += blockDim.x) {
            int r = idx / N, c = idx % N;
            gM_s[idx] -= tmp_s[r] * V_s[k*N+c];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < NN; idx += blockDim.x) {
        int r = idx / N, c = idx % N;
        float s = 0.0f;
        for (int j = 0; j < N; j++)
            s += B_s[r*N+j] * (gM_s[j*N+c] + gM_s[c*N+j]);
        gB_s[idx] += s;
    }
    __syncthreads();

    for (int k = tid; k < n_biv; k += blockDim.x) {
        int ii = skew_i[k], jj = skew_j[k];
        grad_biv_out[b * n_biv + k] = 2.0f * (gB_s[ii*N+jj] - gB_s[jj*N+ii]);
    }
}

std::vector<torch::Tensor> eigh_exp_fwd(
    torch::Tensor bivecs,
    torch::Tensor skew_i,
    torch::Tensor skew_j,
    int N, int K)
{
    TORCH_CHECK(bivecs.is_cuda() && bivecs.is_contiguous());
    const at::cuda::CUDAGuard device_guard(bivecs.device());

    int Bp = bivecs.size(0);
    int n_biv = bivecs.size(1);
    int NN = N * N;

    auto skew_i32 = skew_i.to(torch::kInt32);
    auto skew_j32 = skew_j.to(torch::kInt32);

    auto mat = torch::zeros({Bp, N, N}, bivecs.options());
    mat.index_put_({torch::indexing::Slice(), skew_i.to(torch::kLong), skew_j.to(torch::kLong)}, bivecs);
    auto B = 2.0f * (mat - mat.transpose(-1, -2));

    auto neg_B2 = B.transpose(-1, -2).bmm(B);

    torch::Tensor eigenvalues, Q;
    std::tie(eigenvalues, Q) = torch::linalg_eigh(neg_B2);
    Q = Q.contiguous();

    auto eigenvalues_clean = eigenvalues.clone();
    for (int i = 0; i < K; i++) {
        int idx_a = N - 1 - 2*i;
        int idx_b = N - 1 - 2*i - 1;
        eigenvalues_clean.select(1, idx_b).copy_(eigenvalues_clean.select(1, idx_a));
    }

    auto pick_idx_cpu = torch::empty({K}, torch::kInt32);
    for (int i = 0; i < K; i++)
        pick_idx_cpu[i] = N - 1 - 2*i;
    auto pick_idx = pick_idx_cpu.to(bivecs.device());

    auto R = torch::empty({Bp, N, N}, bivecs.options());
    auto V_saved = torch::empty({Bp, K, N}, bivecs.options());
    auto phi_saved = torch::empty({Bp, K}, bivecs.options());

    int threads = N;
    int smem = (2*NN + 2*K*N + K) * sizeof(float);

    closed_form_R_fwd_kernel<<<Bp, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
        B.data_ptr<float>(), eigenvalues.data_ptr<float>(), Q.data_ptr<float>(),
        R.data_ptr<float>(), V_saved.data_ptr<float>(), phi_saved.data_ptr<float>(),
        pick_idx.data_ptr<int>(), N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {R, B, V_saved, phi_saved, eigenvalues_clean, Q, pick_idx, skew_i32, skew_j32};
}


torch::Tensor eigh_exp_bwd(
    torch::Tensor grad_R,
    torch::Tensor B,
    torch::Tensor V_saved,
    torch::Tensor phi_saved,
    torch::Tensor eigenvalues_clean,
    torch::Tensor Q,
    torch::Tensor skew_i32,
    torch::Tensor skew_j32,
    torch::Tensor pick_idx,
    int N, int K, int n_biv)
{
    TORCH_CHECK(grad_R.is_cuda());
    TORCH_CHECK(skew_i32.scalar_type() == torch::kInt32);
    const at::cuda::CUDAGuard device_guard(grad_R.device());

    int Bp = grad_R.size(0);
    auto grad_biv = torch::empty({Bp, n_biv}, grad_R.options());

    int threads = N;
    int smem = (5*N*N + 4*K*N + 2*K + 2*N) * sizeof(float);

    eigh_exp_bwd_kernel<<<Bp, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
        grad_R.contiguous().data_ptr<float>(),
        B.data_ptr<float>(),
        V_saved.data_ptr<float>(),
        phi_saved.data_ptr<float>(),
        eigenvalues_clean.data_ptr<float>(),
        Q.data_ptr<float>(),
        skew_i32.data_ptr<int>(),
        skew_j32.data_ptr<int>(),
        grad_biv.data_ptr<float>(),
        N, K, n_biv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_biv;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eigh_exp_fwd", &eigh_exp_fwd,
          "eigh + closed-form exp(B) forward");
    m.def("eigh_exp_bwd", &eigh_exp_bwd,
          "eigh + closed-form exp(B) backward");
}
