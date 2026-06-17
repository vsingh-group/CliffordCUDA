"""Python wrapper for the Cl(p, q, r) inner (Hestenes) product CUDA kernel.

Forward bakes the Hestenes predicate `(i ⊆ j) OR (j ⊆ i)` into the
geom_prod packed-sign / packed-valid LUTs (`build_inner_sign_fwd`).
Backward uses dedicated σ_inner row/column LUTs (`build_inner_sign_bwd`)
through the same packed backward kernel as geom_prod — see
`_InnerProdFunc` below for the exact chain-rule sums.
"""
import functools
import os

import torch

from ...._config import _GA_KERNELS_DIR, load_extension
from ..geom_prod import (
    _build_sign_value_table, _normalize_metric, build_packed_sign,
    pack_bwd_from_sigma, pack_fwd_from_sigma,
)


def _build_inner_sigma(n: int, metric=None) -> torch.Tensor:
    """σ_inner(i, j) = σ(i, j) · 𝟙[(i⊆j) ∨ (j⊆i)]. Hestenes inner-product
    grade selector. Returned as int8 (dim, dim) on CPU."""
    metric = _normalize_metric(n, metric)
    dim = 1 << n
    sv = _build_sign_value_table(n, metric)
    i_arr = torch.arange(dim, dtype=torch.int64).view(-1, 1).expand(dim, dim)
    j_arr = torch.arange(dim, dtype=torch.int64).view(1, -1).expand(dim, dim)
    mask = ((i_arr & ~j_arr) == 0) | ((j_arr & ~i_arr) == 0)  # i⊆j ∨ j⊆i
    return (sv.to(torch.int64) * mask.to(torch.int64)).to(torch.int8)


@functools.lru_cache(maxsize=None)
def build_inner_sign_bwd(n: int, device: str = 'cuda', metric=None):
    """Backward LUTs for inner_prod (Hestenes)."""
    if n < 5:
        raise ValueError("n>=5 required")
    return pack_bwd_from_sigma(_build_inner_sigma(n, metric), device)


@functools.lru_cache(maxsize=None)
def build_inner_sign_fwd(n: int, device: str = 'cuda', metric=None):
    """Forward LUT for inner through the GP kernel: σ_inner baked into the
    (packed_sign, packed_valid) pair so geom_prod_fwd[_multik] computes the
    inner product with the predicate masked via packed_valid."""
    if n < 5:
        raise ValueError("n>=5 required")
    return pack_fwd_from_sigma(_build_inner_sigma(n, metric), device)


def load_inner_prod_cuda():
    if not hasattr(load_inner_prod_cuda, '_module'):
        load_inner_prod_cuda._module = load_extension(
            name='inner_prod_cuda',
            sources=[os.path.join(_GA_KERNELS_DIR, 'inner_prod.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_inner_prod_cuda._module


def _inner_raw(a, b, ps, pv):
    return load_inner_prod_cuda().inner_prod_fwd(a, b, ps, pv)


class _InnerProdFunc(torch.autograd.Function):
    """Autograd wrapper for `inner_prod` (Hestenes).

    forward:  c[k] = Σ_{(i,j): i⊆j ∨ j⊆i, i^j=k} σ(i, j) · a[i] · b[j]
    backward: grad_a[k] = Σ_i σ_inner(k, i) · b[i] · grad_c[k^i]
              grad_b[k] = Σ_i σ_inner(i, k) · a[i] · grad_c[k^i]

    σ_inner(i, j) = σ(i, j) · 𝟙[(i⊆j) ∨ (j⊆i)]. Both backward sums have the
    GP forward kernel shape — invoked via geom_prod_fwd with inner-specific
    bwd LUTs (`build_inner_sign_bwd`). Works for any Cl(p, q, r) including
    degenerate.
    """

    @staticmethod
    def forward(ctx, a, b, ps, pv, n, metric_key, use_skip):
        ctx.save_for_backward(a, b)
        ctx.n = n
        ctx.metric_key = metric_key
        ctx.use_skip = use_skip
        return _inner_raw(a, b, ps, pv)

    @staticmethod
    def backward(ctx, grad_c):
        from ..geom_prod import load_geom_prod_cuda
        a, b = ctx.saved_tensors
        dev = str(grad_c.device)
        ps_a, pv_a, ps_b, pv_b = build_inner_sign_bwd(ctx.n, dev, ctx.metric_key)
        gp = load_geom_prod_cuda()
        bwd = gp.geom_prod_fwd_skip if ctx.use_skip else gp.geom_prod_fwd
        grad_c = grad_c.contiguous()
        grad_a = bwd(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = bwd(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None, None


def inner_prod(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Compute c = a · b (Hestenes inner) in Cl(p, q, r) under bit-pattern
    blade indexing. metric=None -> Cl(n, 0). Differentiable via _InnerProdFunc."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_packed_sign(n, str(a.device), metric_key)
    return _InnerProdFunc.apply(a, b, ps, pv, n, metric_key, False)


def inner_prod_skip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Same op as inner_prod, but backward routes through geom_prod_fwd_skip
    (chunk-skip). σ_inner is denser than σ_wedge — the skip variant usually
    regresses on small n for inner. Kept for benchmarking."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_packed_sign(n, str(a.device), metric_key)
    return _InnerProdFunc.apply(a, b, ps, pv, n, metric_key, True)


def inner_prod_kskip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Inner product forward via the `inner_prod_fwd_skip` kernel — the
    predicate-aware (address-only) chunk-skip variant of inner_prod_fwd.
    Forward-only call (no autograd attached) — meant for the forward bench
    column. Bit-identical output to inner_prod's forward."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_packed_sign(n, str(a.device), metric_key)
    return load_inner_prod_cuda().inner_prod_fwd_skip(a, b, ps, pv)


class _InnerProdMultikFunc(torch.autograd.Function):
    """Autograd wrapper for `inner_prod_multik`.

    Forward and backward both go through `geom_prod_fwd_multik` with
    inner-specific LUTs (`build_inner_sign_fwd` / `build_inner_sign_bwd`).
    Same chain-rule derivation as `_InnerProdFunc`, just routed through the
    multik kernel. Works for any Cl(p, q, r) including degenerate.
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
        ps_a, pv_a, ps_b, pv_b = build_inner_sign_bwd(ctx.n, dev, ctx.metric_key)
        gp = load_geom_prod_cuda()
        bwd = gp.geom_prod_fwd_multik_skip if ctx.use_skip else gp.geom_prod_fwd_multik
        grad_c = grad_c.contiguous()
        grad_a = bwd(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = bwd(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None, None


def inner_prod_multik(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Inner (Hestenes) via the GP multik kernel — the inner predicate is
    baked into the forward LUT (`build_inner_sign_fwd`) so a single multik
    launch produces each warp's M=2 outputs. Differentiable via
    `_InnerProdMultikFunc`. Works for any Cl(p, q, r) including degenerate."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_inner_sign_fwd(n, str(a.device), metric_key)
    return _InnerProdMultikFunc.apply(a, b, ps, pv, n, metric_key, False)


def inner_prod_multik_skip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Same op as inner_prod_multik, but uses geom_prod_fwd_multik_skip."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_inner_sign_fwd(n, str(a.device), metric_key)
    return _InnerProdMultikFunc.apply(a, b, ps, pv, n, metric_key, True)
