import os
import torch
from ..._config import _ROTOR_KERNELS_DIR, load_extension


def load_eigh_exp_cuda():
    if not hasattr(load_eigh_exp_cuda, '_module'):
        load_eigh_exp_cuda._module = load_extension(
            name='eigh_exp_cuda',
            sources=[os.path.join(_ROTOR_KERNELS_DIR, 'eigh_exp.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_eigh_exp_cuda._module


class EighExpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bivecs, skew_i, skew_j, N, K):
        results = load_eigh_exp_cuda().eigh_exp_fwd(bivecs.contiguous(), skew_i, skew_j, N, K)
        R, B, V_saved, phi_saved, ec, Q, pick_idx, si32, sj32 = results
        ctx.save_for_backward(B, V_saved, phi_saved, ec, Q, si32, sj32, pick_idx)
        ctx.N = N
        ctx.K = K
        ctx.n_biv = bivecs.size(1)
        return R

    @staticmethod
    def backward(ctx, grad_R):
        B, V_saved, phi_saved, ec, Q, si32, sj32, pick_idx = ctx.saved_tensors
        grad_biv = load_eigh_exp_cuda().eigh_exp_bwd(
            grad_R.contiguous(), B, V_saved, phi_saved,
            ec, Q, si32, sj32, pick_idx,
            ctx.N, ctx.K, ctx.n_biv)
        return grad_biv, None, None, None, None
