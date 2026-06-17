import os
import torch
from ...._config import _ROTOR_KERNELS_DIR, load_extension


def load_givens_apply_mb_packed_cuda():
    if not hasattr(load_givens_apply_mb_packed_cuda, '_module'):
        load_givens_apply_mb_packed_cuda._module = load_extension(
            name='givens_apply_mb_packed_cuda',
            sources=[os.path.join(_ROTOR_KERNELS_DIR, 'givens_apply_mb_packed.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_givens_apply_mb_packed_cuda._module


from . import pack_indices  # shared packer (re-exported for callers)


class GivensApplyMbPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cs_tensor, packed_ij, packed_sig, K, M):
        mod = load_givens_apply_mb_packed_cuda()
        x_out = mod.givens_mb_packed_fwd(
            x.contiguous(), cs_tensor.contiguous(),
            packed_ij, packed_sig, K, M)
        torch.cuda.synchronize(x.device)
        ctx.save_for_backward(x_out, cs_tensor)
        ctx.packed_ij, ctx.packed_sig = packed_ij, packed_sig
        ctx.K, ctx.M = K, M
        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        x_out, cs_tensor = ctx.saved_tensors
        mod = load_givens_apply_mb_packed_cuda()
        grad_cs = mod.givens_mb_packed_bwd(
            grad_output.contiguous(), x_out.contiguous(),
            cs_tensor.contiguous(), ctx.packed_ij, ctx.packed_sig,
            ctx.K, ctx.M)
        torch.cuda.synchronize(grad_output.device)
        return None, grad_cs, None, None, None, None
