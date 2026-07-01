"""Python wrapper for the wedge (exterior) product CUDA kernel.

Reuses the geom_prod packed sign LUT — for wedge, sigma_val reduces to the
reorder parity (no shared bases, so the metric factor product is empty), so
the LUT is signature-independent. The kernel itself zeros out non-wedge terms
via the i ⊆ k predicate, so the same kernel works in any Cl(p, q, r).

Forward is metric-independent. Backward needs the metric for the
metric-weighted adjoint and the contractions it dispatches to.
"""
import functools
import os

import torch

from ...._config import _GA_KERNELS_DIR, load_extension
from ..geom_prod import (
    _build_sign_value_table, _normalize_metric, build_packed_sign,
    pack_bwd_from_sigma, pack_fwd_from_sigma,
)


def _build_wedge_sigma(n: int, metric=None) -> torch.Tensor:
    """σ_wedge(i, j) = σ(i, j) · 𝟙[i & j == 0]. Disjoint-blade mask makes the
    metric factor product empty, so the mask itself is metric-independent;
    degenerate generators are still respected via the underlying σ table.
    Returned as int8 (dim, dim) on CPU."""
    metric = _normalize_metric(n, metric)
    dim = 1 << n
    sv = _build_sign_value_table(n, metric)
    i_arr = torch.arange(dim, dtype=torch.int64).view(-1, 1).expand(dim, dim)
    j_arr = torch.arange(dim, dtype=torch.int64).view(1, -1).expand(dim, dim)
    mask = ((i_arr & j_arr) == 0)
    return (sv.to(torch.int64) * mask.to(torch.int64)).to(torch.int8)


@functools.lru_cache(maxsize=None)
def build_wedge_sign_bwd(n: int, device: str = 'cuda', metric=None):
    """Backward LUTs for wedge."""
    if n < 5:
        raise ValueError("n>=5 required")
    return pack_bwd_from_sigma(_build_wedge_sigma(n, metric), device)


@functools.lru_cache(maxsize=None)
def build_wedge_sign_fwd(n: int, device: str = 'cuda', metric=None):
    """Forward LUT for wedge through the GP kernel: σ_w baked into the
    (packed_sign, packed_valid) pair so geom_prod_fwd[_multik] computes the
    wedge with the predicate masked via packed_valid (HAS_ZEROS=true path)."""
    if n < 5:
        raise ValueError("n>=5 required")
    return pack_fwd_from_sigma(_build_wedge_sigma(n, metric), device)


def load_wedge_prod_cuda():
    if not hasattr(load_wedge_prod_cuda, '_module'):
        load_wedge_prod_cuda._module = load_extension(
            name='wedge_prod_cuda',
            sources=[os.path.join(_GA_KERNELS_DIR, 'wedge_prod.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_wedge_prod_cuda._module


def _wedge_raw(a, b, ps):
    return load_wedge_prod_cuda().wedge_prod_fwd(a, b, ps)


class _WedgeProdFunc(torch.autograd.Function):
    """Autograd wrapper for `wedge_prod`.

    forward:  c[k] = Σ_{i⊆k} ε(i, k\\i) · a[i] · b[k\\i]
    backward: grad_a[k] = Σ_i σ_w(k, i) · b[i] · grad_c[k^i]   (row k of σ_w)
              grad_b[k] = Σ_i σ_w(i, k) · a[i] · grad_c[k^i]   (col k of σ_w)

    Both sums have the GP forward kernel shape — invoked via geom_prod_fwd
    with wedge-specific bwd LUTs (`build_wedge_sign_bwd`). Works for any
    Cl(p, q, r) including degenerate.
    """

    @staticmethod
    def forward(ctx, a, b, ps, n, metric_key, use_skip):
        ctx.save_for_backward(a, b)
        ctx.n = n
        ctx.metric_key = metric_key
        ctx.use_skip = use_skip
        return _wedge_raw(a, b, ps)

    @staticmethod
    def backward(ctx, grad_c):
        from ..geom_prod import load_geom_prod_cuda

        a, b = ctx.saved_tensors
        dev = str(grad_c.device)
        ps_a, pv_a, ps_b, pv_b = build_wedge_sign_bwd(ctx.n, dev, ctx.metric_key)
        gp = load_geom_prod_cuda()
        bwd = gp.geom_prod_fwd_skip if ctx.use_skip else gp.geom_prod_fwd
        grad_c = grad_c.contiguous()
        grad_a = bwd(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = bwd(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None


def wedge_prod(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Compute c = a ∧ b under bit-pattern blade indexing.

    Forward is signature-independent. Backward uses build_wedge_sign_bwd's
    direct-sigma LUTs and works for any Cl(p, q, r) including degenerate."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    if a.is_cuda:                                       # tile past the standard kernel's device limit
        from ..geom_prod_tiled import _STD_KERNEL_MAX_N, _device_max_fit, wedge_prod_tiled_fused
        if n > min(_STD_KERNEL_MAX_N, _device_max_fit(str(a.device), 0 in metric_key)):
            return wedge_prod_tiled_fused(a, b, metric=metric)
    ps, _ = build_packed_sign(n, str(a.device), None)  # forward LUT is signature-free
    return _WedgeProdFunc.apply(a, b, ps, n, metric_key, False)


def wedge_prod_skip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Same op as wedge_prod, but backward routes through geom_prod_fwd_skip
    (chunk-skip enabled). Faster at compute-bound sizes; can regress on small
    n. Forward is unchanged."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, _ = build_packed_sign(n, str(a.device), None)
    return _WedgeProdFunc.apply(a, b, ps, n, metric_key, True)


class _WedgeProdMultikFunc(torch.autograd.Function):
    """Autograd wrapper for `wedge_prod_multik`.

    Forward and backward both go through `geom_prod_fwd_multik` with
    wedge-specific LUTs (`build_wedge_sign_fwd` / `build_wedge_sign_bwd`).
    Same chain-rule derivation as `_WedgeProdFunc`, just routed through the
    multik kernel (each warp emits M=2 outputs, sharing the per-chunk operand
    load). Works for any Cl(p, q, r) including degenerate.
    """

    @staticmethod
    def forward(ctx, a, b, ps, pv, n, metric_key, use_skip):
        from ..geom_prod import load_geom_prod_cuda
        ctx.save_for_backward(a, b)
        ctx.n = n
        ctx.metric_key = metric_key
        ctx.use_skip = use_skip
        gp = load_geom_prod_cuda()
        fwd = gp.geom_prod_fwd_multik_skip if use_skip else gp.geom_prod_fwd_multik
        return fwd(a, b, ps, pv)

    @staticmethod
    def backward(ctx, grad_c):
        from ..geom_prod import load_geom_prod_cuda
        a, b = ctx.saved_tensors
        dev = str(grad_c.device)
        ps_a, pv_a, ps_b, pv_b = build_wedge_sign_bwd(ctx.n, dev, ctx.metric_key)
        gp = load_geom_prod_cuda()
        bwd = gp.geom_prod_fwd_multik_skip if ctx.use_skip else gp.geom_prod_fwd_multik
        grad_c = grad_c.contiguous()
        grad_a = bwd(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = bwd(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None, None


def wedge_prod_multik(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Wedge via the GP multik kernel — the wedge predicate is baked into the
    forward LUT (`build_wedge_sign_fwd`) so a single multik launch produces
    each warp's M=2 outputs. Differentiable via `_WedgeProdMultikFunc`. Works
    for any Cl(p, q, r) including degenerate."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_wedge_sign_fwd(n, str(a.device), metric_key)
    return _WedgeProdMultikFunc.apply(a, b, ps, pv, n, metric_key, False)


def wedge_prod_multik_skip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Same op as wedge_prod_multik, but uses geom_prod_fwd_multik_skip
    (chunk-skip on the multik kernel's HAS_ZEROS path)."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_wedge_sign_fwd(n, str(a.device), metric_key)
    return _WedgeProdMultikFunc.apply(a, b, ps, pv, n, metric_key, True)
