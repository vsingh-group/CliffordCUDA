"""Regressive (meet) product `A ∨ B` for Cl(p, q) — non-degenerate signatures.

Direct path: σ_reg LUT built from the closed-form dual ∘ wedge ∘ dual fusion,
then a single geom_prod_fwd kernel call (+ one index_select to undo the
output dual permutation). Backward goes through two more geom_prod_fwd kernel
calls with σ_reg bwd LUTs — same shape as wedge_prod / inner_prod / contracts.

Derivation: result[k] = ds[k] · wedge(D(a), D(b))[~k]. Reindex by k' = ~k and
substitute p = ~i (wedge first-operand index):

    f[k'] := result[~k']
           = Σ_{p: ~p ⊆ p^k'} σ_reg(p, p^k') · a[p] · b[p^k']

    σ_reg(p, q) = ds[~(p^q)] · σ_wedge(~p, p AND ~q) · ds[~p] · ds[p AND ~q]
                  if  p OR q == FULL,   else 0.

Then result[k] = f[~k] (a single index_select).

regressive_prod_subset_grade still goes through dual → wedge_subset → dual
(subset_grade has no autograd backward yet); kept as the second benchmark
variant.

The regressive product is undefined when the pseudoscalar is non-invertible
(i.e. when the metric has any degenerate generator, r > 0). This module
raises in those cases.
"""
import functools

import torch

from .geom_prod import (
    _build_sign_value_table, _normalize_metric, load_geom_prod_cuda,
    pack_bwd_from_sigma, pack_fwd_from_sigma,
)
from .wedge_prod.subset_grade import wedge_prod_subset_grade


@functools.lru_cache(maxsize=None)
def _build_dual(n: int, device: str = "cuda", metric=None):
    """Returns (dual_signs, dual_perm) such that
        dual(x)[..., i] = dual_signs[i] * x[..., dual_perm[i]]
    implements A* = A · I^{-1} for Cl(p, q) (non-degenerate).

    See module docstring for derivation.
    """
    if n < 5:
        raise ValueError("n>=5 required")
    metric = _normalize_metric(n, metric)
    if 0 in metric:
        raise ValueError(
            f"regressive_prod requires a non-degenerate metric "
            f"(no zero entries); got metric={metric}. The pseudoscalar has "
            f"I·I = 0 in degenerate signatures, so I^{{-1}} does not exist."
        )

    dim = 1 << n
    full = dim - 1
    sigma = _build_sign_value_table(n, metric)              # (dim, dim) int8
    I_sq = int(sigma[full, full])
    assert I_sq in (-1, 1), f"non-degenerate metric should have I·I = ±1, got {I_sq}"

    idx = torch.arange(dim, dtype=torch.long)
    neg_idx = (~idx) & full
    dual_signs = I_sq * sigma[neg_idx, full].to(torch.float32)
    dual_perm = neg_idx.to(torch.int64)

    dual_signs = dual_signs.to(device=device).contiguous()
    dual_perm  = dual_perm.to(device=device).contiguous()
    return dual_signs, dual_perm


def _dual(x: torch.Tensor, dual_signs: torch.Tensor, dual_perm: torch.Tensor) -> torch.Tensor:
    """Apply dual: dual(x)[..., i] = dual_signs[i] * x[..., dual_perm[i]]."""
    return dual_signs * x.index_select(-1, dual_perm)


def _build_regressive_sigma(n: int, metric) -> torch.Tensor:
    """σ_reg(p, q) table for the direct regressive kernel (see module docstring
    for derivation). Returned int8 (dim, dim) on CPU with values in {-1, 0, 1};
    non-zero only where p OR q == FULL (predicate `~p ⊆ q`)."""
    metric = _normalize_metric(n, metric)
    if 0 in metric:
        raise ValueError(
            "direct regressive σ_reg undefined for degenerate metrics (I·I = 0)"
        )

    dim = 1 << n
    full = dim - 1
    sigma = _build_sign_value_table(n, metric).to(torch.int64)
    I_sq = int(sigma[full, full])

    idx = torch.arange(dim, dtype=torch.long)
    neg_idx = (~idx) & full
    # ds[i] = I_sq * sigma[~i, full]
    ds = (I_sq * sigma[neg_idx, full]).to(torch.int64)        # (dim,) in {-1, +1}

    p_arr = torch.arange(dim, dtype=torch.long).view(-1, 1).expand(dim, dim)
    q_arr = torch.arange(dim, dtype=torch.long).view(1, -1).expand(dim, dim)
    not_p = (~p_arr) & full
    not_q = (~q_arr) & full

    # Predicate: p OR q == FULL  ⟺  ~p ⊆ q  ⟺  ~p & ~q == 0
    pred = ((not_p & not_q) == 0)

    # Helper indices.
    k_prime = p_arr ^ q_arr                # = ~k (output-dual reindex)
    not_k_prime = (~k_prime) & full        # = k
    p_and_notq = p_arr & not_q             # = p AND ~q ; under predicate, equals p\k' = p AND k'

    # σ_wedge(~p, p AND ~q): when the predicate holds the two args are disjoint,
    # so this equals the pure reorder parity (metric factor product over an
    # empty intersection is 1). We still pull from the full σ table so the row
    # values are correct (the predicate mask zeros out the rest).
    sigma_wedge = sigma[not_p.reshape(-1), p_and_notq.reshape(-1)].view(dim, dim)

    sigma_reg = (ds[not_k_prime.reshape(-1)].view(dim, dim)
                 * sigma_wedge
                 * ds[not_p.reshape(-1)].view(dim, dim)
                 * ds[p_and_notq.reshape(-1)].view(dim, dim))
    sigma_reg = sigma_reg * pred.to(torch.int64)
    return sigma_reg.to(torch.int8)


@functools.lru_cache(maxsize=None)
def build_regressive_sign_fwd(n: int, device: str = "cuda", metric=None):
    """Forward LUT for the direct regressive geom_prod_fwd call. The σ_reg
    table is dense with zeros (most (p, q) pairs fail the predicate), so
    pack_fwd_from_sigma builds a packed_valid LUT for the HAS_ZEROS path."""
    if n < 5:
        raise ValueError("n>=5 required")
    return pack_fwd_from_sigma(_build_regressive_sigma(n, metric), device)


@functools.lru_cache(maxsize=None)
def build_regressive_sign_bwd(n: int, device: str = "cuda", metric=None):
    """Backward LUTs for the direct regressive backward pass via the
    direct-σ-LUT chain rule (row k of σ_reg → grad_a, col k → grad_b)."""
    if n < 5:
        raise ValueError("n>=5 required")
    return pack_bwd_from_sigma(_build_regressive_sigma(n, metric), device)


class _RegressiveProdFunc(torch.autograd.Function):
    """Autograd wrapper for direct regressive_prod.

    forward:  f[k'] = Σ_p σ_reg(p, p^k') · a[p] · b[p^k']    (geom_prod_fwd)
              result[k] = f[~k]                              (index_select)

    backward: grad_f[k']   = grad_result[~k']                (index_select)
              grad_a[k]    = Σ_i σ_reg(k, i) · b[i] · grad_f[k^i]   (row k of σ_reg)
              grad_b[k]    = Σ_i σ_reg(i, k) · a[i] · grad_f[k^i]   (col k of σ_reg)

    1 GP kernel + 1 gather forward; 1 gather + 2 GP kernels backward — same
    shape as wedge_prod / inner_prod / contract autograd wrappers. Eliminates
    the launch-overhead floor the previous dual→wedge→dual autograd
    composition incurred.
    """

    @staticmethod
    def forward(ctx, a, b, ps, pv, dp, n, metric_key, use_skip):
        ctx.save_for_backward(a, b)
        ctx.dp = dp
        ctx.n = n
        ctx.metric_key = metric_key
        ctx.use_skip = use_skip
        gp = load_geom_prod_cuda()
        fwd = gp.geom_prod_fwd_skip if use_skip else gp.geom_prod_fwd
        f = fwd(a, b, ps, pv)
        return f.index_select(-1, dp).contiguous()

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        dp = ctx.dp
        dev = str(grad_out.device)
        grad_f = grad_out.contiguous().index_select(-1, dp).contiguous()
        ps_a, pv_a, ps_b, pv_b = build_regressive_sign_bwd(
            ctx.n, dev, ctx.metric_key)
        gp = load_geom_prod_cuda()
        bwd = gp.geom_prod_fwd_skip if ctx.use_skip else gp.geom_prod_fwd
        grad_a = bwd(b.contiguous(), grad_f, ps_a, pv_a)
        grad_b = bwd(a.contiguous(), grad_f, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None, None, None


def regressive_prod(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Regressive product a ∨ b in Cl(p, q). Direct path: single GP kernel
    forward (+ one index_select) and two GP kernels for backward."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    if a.is_cuda:                                       # tile past the standard kernel's device limit
        from .geom_prod_tiled import _STD_KERNEL_MAX_N, _device_max_fit, regressive_prod_tiled_fused
        if n > min(_STD_KERNEL_MAX_N, _device_max_fit(str(a.device), 0 in metric_key)):
            return regressive_prod_tiled_fused(a, b, metric=metric)
    ps, pv = build_regressive_sign_fwd(n, str(a.device), metric_key)
    _ds, dp = _build_dual(n, str(a.device), metric_key)
    return _RegressiveProdFunc.apply(a, b, ps, pv, dp, n, metric_key, False)


def regressive_prod_skip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Same op as regressive_prod, but both forward and backward go through
    geom_prod_fwd_skip (chunk-skip enabled). σ_reg is sparse, so the skip
    typically wins at compute-bound sizes."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_regressive_sign_fwd(n, str(a.device), metric_key)
    _ds, dp = _build_dual(n, str(a.device), metric_key)
    return _RegressiveProdFunc.apply(a, b, ps, pv, dp, n, metric_key, True)


def regressive_prod_subset_grade(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Subset-grade variant — kept on the legacy dual → wedge_subset → dual
    composition (wedge_prod_subset_grade has no autograd backward yet, so
    this path is forward-only). Used by the subset benchmark column."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ds, dp = _build_dual(n, str(a.device), metric_key)
    a_dual = _dual(a, ds, dp)
    b_dual = _dual(b, ds, dp)
    c_wedge = wedge_prod_subset_grade(a_dual, b_dual)
    return _dual(c_wedge, ds, dp)
