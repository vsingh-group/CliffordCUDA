// =============================================================================
// Cl(n, 0) wedge product — SUBSET-ITERATION + GRADE ORDERING (Idea 1 + 2).
// =============================================================================
//
// One of three wedge implementations:
//   - wedge_prod.cu              (chunk variant)
//   - wedge_prod_subset.cu       (Idea 1 only)
//   - wedge_prod_subset_grade.cu (THIS FILE: Idea 1 + 2)
//
// Same as wedge_prod_subset.cu but the warp -> k mapping goes through
// k_by_grade[my_idx], a permutation of [0, dim) sorted by popcount(k). All W
// warps in a block share the same |k|, so n_iters is uniform per block and
// no warp idles waiting for the slowest one.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


template<int N, int W>
__global__ void wedge_prod_subset_grade_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int*   __restrict__ i_lut,
    const int*   __restrict__ k_offset_i,
    const int*   __restrict__ sign_lut,
    const int*   __restrict__ k_offset_sign,
    const int*   __restrict__ k_by_grade,
    float*       __restrict__ c,
    int batch)
{
    constexpr int DIM = 1 << N;

    int b_idx = blockIdx.x;
    if (b_idx >= batch) return;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int my_idx  = blockIdx.y * W + warp_id;
    int k       = (my_idx < DIM) ? k_by_grade[my_idx] : -1;

    extern __shared__ unsigned char smem_raw[];
    float* sA = reinterpret_cast<float*>(smem_raw);
    float* sB = sA + DIM;

    int blocksize = W * 32;
    #pragma unroll 1
    for (int i = tid; i < DIM; i += blocksize) {
        sA[i] = a[b_idx * DIM + i];
        sB[i] = b[b_idx * DIM + i];
    }
    __syncthreads();
    if (k < 0) return;

    int popk = __popc(k);
    int num_subsets = 1 << popk;
    int n_iters = (num_subsets + 31) >> 5;
    int off_i = k_offset_i[k];
    int off_s = k_offset_sign[k];

    float acc = 0.0f;

    #pragma unroll 1
    for (int w = 0; w < n_iters; w++) {
        int t_global = (w << 5) + lane;
        bool active = (t_global < num_subsets);

        int safe_t = active ? t_global : 0;
        int i_val = i_lut[off_i + safe_t];
        int j_val = k ^ i_val;

        int sign_word = sign_lut[off_s + w];
        float s = 1.0f - 2.0f * float((sign_word >> lane) & 1);

        float a_val = sA[i_val];
        float b_val = sB[j_val];

        acc += float(active) * s * a_val * b_val;
    }

    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, delta);

    if (lane == 0)
        c[b_idx * DIM + k] = acc;
}


#define LAUNCH_SUBSET_GRADE(N_VAL, W_VAL)                                            \
    do {                                                                              \
        constexpr int DIM = 1 << (N_VAL);                                             \
        int blocks_y = (DIM + (W_VAL) - 1) / (W_VAL);                                 \
        dim3 grid(batch, blocks_y);                                                   \
        int threads = (W_VAL) * 32;                                                   \
        int smem = 2 * DIM * (int)sizeof(float);                                      \
        if (smem > 48 * 1024)                                                         \
            TORCH_CHECK(cudaFuncSetAttribute(                                                     \
                wedge_prod_subset_grade_kernel<(N_VAL), (W_VAL)>,                     \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "dynamic SMEM opt-in failed");                   \
        wedge_prod_subset_grade_kernel<(N_VAL), (W_VAL)>                              \
            <<<grid, threads, smem, stream>>>(a_ptr, b_ptr,                           \
                                              i_lut_ptr, koi_ptr,                     \
                                              sign_lut_ptr, kos_ptr,                  \
                                              k_by_grade_ptr,                         \
                                              c_ptr, batch);                          \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } while (0)


torch::Tensor wedge_prod_subset_grade_fwd(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor i_lut, torch::Tensor k_offset_i,
    torch::Tensor sign_lut, torch::Tensor k_offset_sign,
    torch::Tensor k_by_grade)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(i_lut.is_cuda() && i_lut.scalar_type() == torch::kInt32);
    TORCH_CHECK(k_offset_i.is_cuda() && k_offset_i.scalar_type() == torch::kInt32);
    TORCH_CHECK(sign_lut.is_cuda() && sign_lut.scalar_type() == torch::kInt32);
    TORCH_CHECK(k_offset_sign.is_cuda() && k_offset_sign.scalar_type() == torch::kInt32);
    TORCH_CHECK(k_by_grade.is_cuda() && k_by_grade.scalar_type() == torch::kInt32);
    TORCH_CHECK(a.sizes() == b.sizes()); TORCH_CHECK(a.dim() == 2);
    const at::cuda::CUDAGuard guard(a.device());

    int batch = a.size(0);
    int dim = a.size(1);
    int N = 0;
    while ((1 << N) < dim) N++;
    TORCH_CHECK((1 << N) == dim); TORCH_CHECK(N >= 5);

    auto c = torch::empty_like(a);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    const int* i_lut_ptr     = i_lut.data_ptr<int>();
    const int* koi_ptr       = k_offset_i.data_ptr<int>();
    const int* sign_lut_ptr  = sign_lut.data_ptr<int>();
    const int* kos_ptr       = k_offset_sign.data_ptr<int>();
    const int* k_by_grade_ptr = k_by_grade.data_ptr<int>();
    float* c_ptr = c.data_ptr<float>();

    if      (N == 5)  LAUNCH_SUBSET_GRADE(5,  4);
    else if (N == 6)  LAUNCH_SUBSET_GRADE(6,  4);
    else if (N == 7)  LAUNCH_SUBSET_GRADE(7,  4);
    else if (N == 8)  LAUNCH_SUBSET_GRADE(8,  4);
    else if (N == 9)  LAUNCH_SUBSET_GRADE(9,  16);
    else if (N == 10) LAUNCH_SUBSET_GRADE(10, 16);
    else if (N == 11) LAUNCH_SUBSET_GRADE(11, 16);
    else if (N == 12) LAUNCH_SUBSET_GRADE(12, 16);
    else if (N == 13) LAUNCH_SUBSET_GRADE(13, 16);
    else TORCH_CHECK(false, "unsupported N");
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wedge_prod_subset_grade_fwd", &wedge_prod_subset_grade_fwd,
          "Cl(n, 0) wedge product, subset-iter + grade-ordered warps (Idea 1 + 2)",
          pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("i_lut"), pybind11::arg("k_offset_i"),
          pybind11::arg("sign_lut"), pybind11::arg("k_offset_sign"),
          pybind11::arg("k_by_grade"));
}
