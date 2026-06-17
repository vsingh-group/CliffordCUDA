"""Python wrapper for the Cl(p, q, r) geometric product CUDA kernel.

Public API:
  - load_geom_prod_cuda(): compile/load the extension.
  - build_packed_sign(n, device, metric=None): build the packed sign LUT (and
                                  validity LUT for degenerate metrics) for the
                                  given algebra. Cached via @functools.lru_cache.
  - geom_prod(a, b, metric=None): compute c = a * b in Cl(p, q, r). Auto-builds
                                  and caches LUTs for the inferred (n, metric).
                                  metric is a tuple of n values from {-1, 0, 1};
                                  None defaults to Cl(n, 0).
"""
import functools
import os

import torch

from ..._config import _GA_KERNELS_DIR, load_extension


def load_geom_prod_cuda():
    if not hasattr(load_geom_prod_cuda, '_module'):
        load_geom_prod_cuda._module = load_extension(
            name='geom_prod_cuda',
            sources=[os.path.join(_GA_KERNELS_DIR, 'geom_prod.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_geom_prod_cuda._module


def _normalize_metric(n, metric):
    """Return a length-n tuple of {-1, 0, 1}. metric=None -> Cl(n, 0)."""
    if metric is None:
        return (1,) * n
    metric = tuple(int(m) for m in metric)
    if len(metric) != n:
        raise ValueError(f"metric must have {n} entries, got {len(metric)}")
    if any(m not in (-1, 0, 1) for m in metric):
        raise ValueError(f"metric entries must be in {{-1, 0, 1}}, got {metric}")
    return metric


def _build_sign_value_table(n: int, metric) -> torch.Tensor:
    """sigma_val[i, j] in {-1, 0, +1}: full geometric-product sign of e_i * e_j
    for the given metric. (i, j) are bit-pattern blade indices.

    sigma_val = (-1)^reorder_parity * prod_{k in bits(i & j)} metric[k]
    where reorder_parity = |{(a, b): a > b, bit_a(i)=1, bit_b(j)=1}|.
    For Cl(n, 0): identical to the 1-bit reordering parity sign.
    Returned as int8 (dim, dim) on CPU."""
    dim = 1 << n
    i_grid = torch.arange(dim, dtype=torch.int64).view(dim, 1).expand(dim, dim)
    j_grid = torch.arange(dim, dtype=torch.int64).view(1, dim).expand(dim, dim)
    parity = torch.zeros(dim, dim, dtype=torch.int64)
    for a in range(n):
        bit_a_i = (i_grid >> a) & 1
        for b in range(a):
            bit_b_j = (j_grid >> b) & 1
            parity += bit_a_i * bit_b_j
    sign = (1 - 2 * (parity & 1)).to(torch.int64)   # ±1 reordering sign
    # Apply metric factors over shared bases (i & j).
    common = i_grid & j_grid
    factor = torch.ones(dim, dim, dtype=torch.int64)
    for k in range(n):
        m = int(metric[k])
        if m == 1:
            continue                          # no-op
        mask = ((common >> k) & 1).to(torch.bool)
        if m == -1:
            factor = torch.where(mask, -factor, factor)
        else:                                 # m == 0
            factor = torch.where(mask, torch.zeros_like(factor), factor)
    return (sign * factor).to(torch.int8)     # values in {-1, 0, +1}


def pack_fwd_from_sigma(sigma_table: torch.Tensor, device: str = "cuda"):
    """Pack forward LUT for the GP kernel from a full (dim, dim) sigma table.

    sigma_table[i, j] in {-1, 0, +1}: GP uses the unmasked σ table; op variants
    (wedge_prod_multik, inner_prod_multik) bake their predicate into σ before
    calling (so the same GP kernel runs the wedge/inner forward through a
    different LUT). Returns (packed_sign, packed_valid_or_None).

      packed_sign[k, c]: int32; bit t = 1 iff sigma_table[c*32+t, (c*32+t)^k] == -1
      packed_valid:      None when the table has no zeros (HAS_ZEROS=false fast
                         path); otherwise int32 with bit t = 1 iff that entry != 0.
    """
    dim = sigma_table.shape[0]
    chunks = dim // 32
    sigma_neg = (sigma_table == -1).to(torch.int32)
    sigma_nz  = (sigma_table != 0).to(torch.int32)
    has_zeros = bool((sigma_table == 0).any().item())
    powers = (1 << torch.arange(32, dtype=torch.int32))

    packed_sign  = torch.empty(dim, chunks, dtype=torch.int32)
    packed_valid = torch.empty(dim, chunks, dtype=torch.int32) if has_zeros else None
    i_arr = torch.arange(dim, dtype=torch.int64)
    for k in range(dim):
        j_arr = i_arr ^ k
        sn_k = sigma_neg[i_arr, j_arr]
        packed_sign[k] = (sn_k.view(chunks, 32) * powers).sum(dim=-1)
        if has_zeros:
            sv_k = sigma_nz[i_arr, j_arr]
            packed_valid[k] = (sv_k.view(chunks, 32) * powers).sum(dim=-1)
    packed_sign = packed_sign.to(device).contiguous()
    if has_zeros:
        packed_valid = packed_valid.to(device).contiguous()
    return packed_sign, packed_valid


def pack_bwd_from_sigma(sigma_table: torch.Tensor, device: str = "cuda"):
    """Pack backward LUTs from a full (dim, dim) sigma table.

    sigma_table: int8 with values in {-1, 0, +1} for any op
    (GP uses full σ, wedge uses σ·𝟙[i&j=0], inner uses σ·𝟙[i⊆j ∨ j⊆i], etc.).

    Returns (bwd_a_sign, bwd_a_valid, bwd_b_sign, bwd_b_valid):
      bwd_a[k, c] packs row k of sigma_table  → encodes σ_op(k, i)
      bwd_b[k, c] packs col k of sigma_table  → encodes σ_op(i, k)
    Both `_valid` LUTs encode `σ_op != 0`. Returned valid LUTs are None when
    the sigma table has no zeros (HAS_ZEROS=false fast path applies).
    """
    dim = sigma_table.shape[0]
    chunks = dim // 32
    sv_neg = (sigma_table == -1).to(torch.int32)
    sv_nz  = (sigma_table != 0).to(torch.int32)
    has_zeros = bool((sigma_table == 0).any().item())
    powers = (1 << torch.arange(32, dtype=torch.int32))

    sv_neg_T = sv_neg.T.contiguous()
    sv_nz_T  = sv_nz.T.contiguous() if has_zeros else None

    bwd_a_sign  = torch.empty(dim, chunks, dtype=torch.int32)
    bwd_b_sign  = torch.empty(dim, chunks, dtype=torch.int32)
    bwd_a_valid = torch.empty(dim, chunks, dtype=torch.int32) if has_zeros else None
    bwd_b_valid = torch.empty(dim, chunks, dtype=torch.int32) if has_zeros else None
    for k in range(dim):
        bwd_a_sign[k] = (sv_neg[k].view(chunks, 32) * powers).sum(dim=-1)
        bwd_b_sign[k] = (sv_neg_T[k].view(chunks, 32) * powers).sum(dim=-1)
        if has_zeros:
            bwd_a_valid[k] = (sv_nz[k].view(chunks, 32) * powers).sum(dim=-1)
            bwd_b_valid[k] = (sv_nz_T[k].view(chunks, 32) * powers).sum(dim=-1)

    bwd_a_sign = bwd_a_sign.to(device).contiguous()
    bwd_b_sign = bwd_b_sign.to(device).contiguous()
    if has_zeros:
        bwd_a_valid = bwd_a_valid.to(device).contiguous()
        bwd_b_valid = bwd_b_valid.to(device).contiguous()
    return bwd_a_sign, bwd_a_valid, bwd_b_sign, bwd_b_valid


@functools.lru_cache(maxsize=None)
def build_packed_sign_bwd(n: int, device: str = 'cuda', metric=None):
    """Backward LUTs for the GP, derived directly from the chain rule.

    Forward:    c[k] = Σ_i σ(i, i^k) · a[i] · b[i^k]
    Backward:   ∂L/∂a[k] = Σ_i σ(k, i)   · b[i] · grad_c[k^i]
                ∂L/∂b[k] = Σ_i σ(i, k)   · a[i] · grad_c[k^i]

    Each backward sum has the same shape as the forward kernel
    (`Σ_i LUT[k,i] · X[i] · Y[i^k]`), just with a different LUT:
      - bwd_a: row k, lane i = sigma_val(k, i)   (row k of full sigma table)
      - bwd_b: row k, lane i = sigma_val(i, k)   (col k of full sigma table)

    Works for any Cl(p, q, r) including degenerate, since `packed_valid`
    already handles σ = 0 terms.

    Returns (bwd_a_sign, bwd_a_valid_or_None, bwd_b_sign, bwd_b_valid_or_None).
    """
    if n < 5:
        raise ValueError("n>=5 required")
    metric = _normalize_metric(n, metric)
    sv = _build_sign_value_table(n, metric)          # (dim, dim) int8 in {-1, 0, +1}
    return pack_bwd_from_sigma(sv, device)


@functools.lru_cache(maxsize=None)
def build_packed_sign(n: int, device: str = 'cuda', metric=None):
    """Forward LUT for the GP kernel in Cl(p, q, r). Returns (packed_sign,
    packed_valid_or_None) by packing the full σ table via `pack_fwd_from_sigma`.

    packed_sign[k, c]: int32 word; bit t = 1 iff sigma_val(c*32+t, (c*32+t)^k) == -1.
    packed_valid:      None for non-degenerate metrics (HAS_ZEROS=false fast path).
                       For degenerate, an int32 LUT of the same shape; bit t = 1
                       iff sigma_val != 0.
    """
    if n < 5:
        raise ValueError("n>=5 required; lower n is not supported by this kernel layout")
    metric = _normalize_metric(n, metric)
    sigma_val = _build_sign_value_table(n, metric)
    return pack_fwd_from_sigma(sigma_val, device)


def _prep(a, b, metric=None):
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, pv = build_packed_sign(n, str(a.device), metric_key)
    return a, b, ps, pv


def _gp_raw(a, b, ps, pv):
    """Raw forward kernel call (no autograd)."""
    return load_geom_prod_cuda().geom_prod_fwd(a, b, ps, pv)


def _gp_raw_multik(a, b, ps, pv):
    """Raw multik forward kernel call (no autograd)."""
    return load_geom_prod_cuda().geom_prod_fwd_multik(a, b, ps, pv)


class _GeomProdFunc(torch.autograd.Function):
    """Autograd wrapper for `geom_prod`.

    Forward:  c[k] = Σ_i σ(i, i^k) · a[i] · b[i^k]   (LUT row k indexes σ(i, i^k))
    Backward: grad_a[k] = Σ_i σ(k, i) · b[i] · grad_c[k^i]   (LUT row k = row k of σ)
              grad_b[k] = Σ_i σ(i, k) · a[i] · grad_c[k^i]   (LUT row k = col k of σ)

    Both backward sums have the same Σ_i LUT[k,i] · X[i] · Y[i^k] shape as the
    forward, so they're just two more `geom_prod_fwd` kernel calls with
    different LUTs (`build_packed_sign_bwd`). Works for any Cl(p, q, r)
    including degenerate metrics — the σ table can have zeros and the
    existing `packed_valid` HAS_ZEROS path handles them.

    Verified by `tests/gradcheck/test_grad_geom_prod.py` against
    torch.autograd.gradcheck across several (p, q, r) signatures, both
    non-degenerate and degenerate.
    """

    @staticmethod
    def forward(ctx, a, b, ps, pv, n, metric_key):
        ctx.save_for_backward(a, b)
        ctx.n = n
        ctx.metric_key = metric_key
        return _gp_raw(a, b, ps, pv)

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        n = ctx.n
        dev = str(grad_c.device)
        ps_a, pv_a, ps_b, pv_b = build_packed_sign_bwd(n, dev, ctx.metric_key)
        grad_c = grad_c.contiguous()
        grad_a = _gp_raw(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = _gp_raw(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None


def geom_prod(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Cl(p, q, r) GP. Default variant: cp.async operands, SMEM-staged sign row,
    K-unroll. metric=None -> Cl(n, 0). Differentiable for any Cl(p, q, r)
    including degenerate (via build_packed_sign_bwd direct-sigma LUTs)."""
    a, b, ps, pv = _prep(a, b, metric)
    dim = a.size(-1); n = dim.bit_length() - 1
    metric_key = _normalize_metric(n, metric)
    return _GeomProdFunc.apply(a, b, ps, pv, n, metric_key)


class _GeomProdMultikFunc(torch.autograd.Function):
    """Autograd wrapper for `geom_prod_multik`.

    Identical math to `_GeomProdFunc` — same chain-rule formulas, same forward
    and backward LUTs — but routes the kernel call through `geom_prod_fwd_multik`
    (each warp emits M=2 output blades, sharing the per-chunk operand load).
    Both forward and backward go through the multik kernel.
    """

    @staticmethod
    def forward(ctx, a, b, ps, pv, n, metric_key):
        ctx.save_for_backward(a, b)
        ctx.n = n
        ctx.metric_key = metric_key
        return _gp_raw_multik(a, b, ps, pv)

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        n = ctx.n
        dev = str(grad_c.device)
        ps_a, pv_a, ps_b, pv_b = build_packed_sign_bwd(n, dev, ctx.metric_key)
        grad_c = grad_c.contiguous()
        grad_a = _gp_raw_multik(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = _gp_raw_multik(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None


def geom_prod_multik(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Variant: each warp produces M=2 output blades, sharing the per-chunk
    operand SMEM read. Differentiable via `_GeomProdMultikFunc` (forward and
    backward both go through the multik kernel)."""
    a, b, ps, pv = _prep(a, b, metric)
    dim = a.size(-1); n = dim.bit_length() - 1
    metric_key = _normalize_metric(n, metric)
    return _GeomProdMultikFunc.apply(a, b, ps, pv, n, metric_key)
