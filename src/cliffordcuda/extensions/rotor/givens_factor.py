import os
import torch
from ..._config import _ROTOR_KERNELS_DIR, load_extension


def load_givens_factor_cuda():
    if not hasattr(load_givens_factor_cuda, '_module'):
        load_givens_factor_cuda._module = load_extension(
            name='givens_factor_cuda',
            sources=[os.path.join(_ROTOR_KERNELS_DIR, 'givens_factor.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_givens_factor_cuda._module


class GivensFactorFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R):
        n = R.size(1)
        n_rot = n * (n - 1) // 2
        cs, A_final = load_givens_factor_cuda().givens_factor_fwd_v2(R.contiguous(), n_rot)
        ctx.save_for_backward(cs, A_final)
        ctx.n_rot = n_rot
        return cs

    @staticmethod
    def backward(ctx, grad_cs):
        cs, A_final = ctx.saved_tensors
        grad_R = load_givens_factor_cuda().givens_factor_bwd_v2(
            cs, A_final, grad_cs.contiguous(), ctx.n_rot)
        return grad_R
