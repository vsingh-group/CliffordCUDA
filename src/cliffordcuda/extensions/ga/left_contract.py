"""Left contraction a ⌋ b in Cl(p, q, r).

  e_i ⌋ e_j = sigma_val(i, j) * e_{j\\i}   if i ⊆ j, else 0.

With output blade k = i^j, the predicate is (i & k) == 0. Shared bases = i.

LUT-driven: the operation is encoded purely in the validity LUT.
  - Chunk variant calls contract_fwd (kernels/contract.cu) — a specialized
    chunk-iter kernel with warp-uniform chunk-skip on `vword == 0`. The
    packed_valid LUT encodes (i & k == 0) AND sigma_val(i, j) != 0.
  - Subset-grade variant reuses inner_prod_subset_grade_fwd. i_lut enumerates
    only x ⊆ comp(k) with no degenerate base in x (x = shared bases of i, j).
"""
import functools
import os

import numpy as np
import torch

from ..._config import _GA_KERNELS_DIR, load_extension
from .geom_prod import (
    _build_sign_value_table, _normalize_metric, build_packed_sign,
    pack_bwd_from_sigma,
)
from .inner_prod.subset_grade import (
    _sigma_val, load_inner_prod_subset_grade_cuda,
)


def load_contract_cuda():
    if not hasattr(load_contract_cuda, '_module'):
        load_contract_cuda._module = load_extension(
            name='contract_cuda',
            sources=[os.path.join(_GA_KERNELS_DIR, 'contract.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_contract_cuda._module


@functools.lru_cache(maxsize=None)
def build_left_contract_valid(n: int, device: str = 'cuda', metric=None) -> torch.Tensor:
    """packed_valid (dim, dim/32) int32. bit t for (k, c) = 1 iff
    (i & k == 0) AND sigma_val(i, i^k) != 0, where i = c*32 + t."""
    if n < 5:
        raise ValueError("n>=5 required")
    metric = _normalize_metric(n, metric)
    dim = 1 << n
    chunks = dim // 32

    sigma_val = _build_sign_value_table(n, metric)
    sigma_nz  = (sigma_val != 0).to(torch.int32)

    powers = (1 << torch.arange(32, dtype=torch.int32))
    packed_valid = torch.empty(dim, chunks, dtype=torch.int32)
    i_arr = torch.arange(dim, dtype=torch.int64)
    for k in range(dim):
        j_arr = i_arr ^ k
        op_pred = ((i_arr & k) == 0).to(torch.int32)
        valid = sigma_nz[i_arr, j_arr] * op_pred
        packed_valid[k] = (valid.view(chunks, 32) * powers).sum(dim=-1)
    return packed_valid.to(device).contiguous()


def _left_raw(a, b, ps, pv):
    return load_contract_cuda().contract_fwd(a, b, ps, pv)


@functools.lru_cache(maxsize=None)
def build_left_contract_sign_bwd(n: int, device: str = 'cuda', metric=None):
    """Backward LUTs for left_contract via direct chain rule.
    σ_lc(i, j) = σ(i, j) · 𝟙[i⊆j]. Predicate is asymmetric, so row-of-σ_lc
    (bwd_a) and col-of-σ_lc (bwd_b) encode different masks, both packed via
    `pack_bwd_from_sigma`. Works for any Cl(p, q, r)."""
    if n < 5:
        raise ValueError("n>=5 required")
    metric = _normalize_metric(n, metric)
    dim = 1 << n
    sv = _build_sign_value_table(n, metric)
    i_arr = torch.arange(dim, dtype=torch.int64).view(-1, 1).expand(dim, dim)
    j_arr = torch.arange(dim, dtype=torch.int64).view(1, -1).expand(dim, dim)
    mask = ((i_arr & ~j_arr) == 0)              # i ⊆ j
    sigma_lc = (sv.to(torch.int64) * mask.to(torch.int64)).to(torch.int8)
    return pack_bwd_from_sigma(sigma_lc, device)


class _LeftContractFunc(torch.autograd.Function):
    """Autograd wrapper for `left_contract`.

    forward:  c[k] = Σ_{i⊆j, j=i^k} σ(i, j) · a[i] · b[j]
    backward: grad_a[k] = Σ_i σ_lc(k, i) · b[i] · grad_c[k^i]   (row k of σ_lc)
              grad_b[k] = Σ_i σ_lc(i, k) · a[i] · grad_c[k^i]   (col k of σ_lc)

    σ_lc(i, j) = σ(i, j) · 𝟙[i⊆j]. Both backward sums have the GP forward
    kernel shape — invoked via geom_prod_fwd with lc-specific bwd LUTs.
    Works for any Cl(p, q, r) including degenerate.
    """

    @staticmethod
    def forward(ctx, a, b, ps, pv, n, metric_key, use_skip):
        ctx.save_for_backward(a, b)
        ctx.n = n
        ctx.metric_key = metric_key
        ctx.use_skip = use_skip
        return _left_raw(a, b, ps, pv)

    @staticmethod
    def backward(ctx, grad_c):
        from .geom_prod import load_geom_prod_cuda
        a, b = ctx.saved_tensors
        dev = str(grad_c.device)
        ps_a, pv_a, ps_b, pv_b = build_left_contract_sign_bwd(ctx.n, dev, ctx.metric_key)
        gp = load_geom_prod_cuda()
        bwd = gp.geom_prod_fwd_skip if ctx.use_skip else gp.geom_prod_fwd
        grad_c = grad_c.contiguous()
        grad_a = bwd(b.contiguous(), grad_c, ps_a, pv_a)
        grad_b = bwd(a.contiguous(), grad_c, ps_b, pv_b)
        return grad_a, grad_b, None, None, None, None, None


def left_contract(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Left contraction `a ⌋ b`. Chunk variant with warp-uniform chunk-skip.
    Differentiable via `_LeftContractFunc` (direct-sigma-LUT chain rule).
    Works for any Cl(p, q, r) including degenerate.
    """
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, _ = build_packed_sign(n, str(a.device), metric_key)
    pv = build_left_contract_valid(n, str(a.device), metric_key)
    return _LeftContractFunc.apply(a, b, ps, pv, n, metric_key, False)


def left_contract_skip(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Same op as left_contract, but backward routes through geom_prod_fwd_skip."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = _normalize_metric(n, metric)
    ps, _ = build_packed_sign(n, str(a.device), metric_key)
    pv = build_left_contract_valid(n, str(a.device), metric_key)
    return _LeftContractFunc.apply(a, b, ps, pv, n, metric_key, True)


@functools.lru_cache(maxsize=None)
def build_left_contract_subset_lut(n: int, device: str = 'cuda', metric=None):
    """For each k: enumerate i = x where x ⊆ comp(k) and (x & deg_mask) == 0.
    The sign LUT encodes full sigma_val (±1 only for surviving terms)."""
    if n < 5:
        raise ValueError("n>=5 required")
    metric = _normalize_metric(n, metric)
    deg_mask = 0
    for k, m in enumerate(metric):
        if m == 0:
            deg_mask |= (1 << k)

    dim = 1 << n
    full_mask = dim - 1

    i_lut_list = []
    sign_lut_list = []
    k_offset_i = [0] * dim
    k_offset_sign = [0] * dim
    num_subsets_lut = [0] * dim
    cum_i = 0
    cum_sign = 0

    for k in range(dim):
        k_offset_i[k] = cum_i
        k_offset_sign[k] = cum_sign

        comp = full_mask & ~k
        comp_bits = [b for b in range(n) if (comp >> b) & 1]
        p = len(comp_bits)
        n_x = 1 << p

        i_vals = []
        for t in range(n_x):
            x = 0
            for b_idx, b_pos in enumerate(comp_bits):
                if (t >> b_idx) & 1:
                    x |= (1 << b_pos)
            if (x & deg_mask) != 0:
                continue
            i_vals.append(x)

        n_subsets = len(i_vals)
        i_lut_list.extend(i_vals)
        num_subsets_lut[k] = n_subsets

        n_iters = (n_subsets + 31) // 32
        for w in range(n_iters):
            word = 0
            for lane in range(32):
                t_global = w * 32 + lane
                if t_global >= n_subsets:
                    continue
                i_val = i_vals[t_global]
                j_val = k ^ i_val
                sv = _sigma_val(i_val, j_val, metric)
                if sv == -1:
                    word |= (1 << lane)
            sign_lut_list.append(word)

        cum_i += n_subsets
        cum_sign += n_iters

    sign_arr = np.array(sign_lut_list, dtype=np.uint32).view(np.int32)
    k_arr = np.arange(dim, dtype=np.int32)
    pop = np.array([bin(int(x)).count('1') for x in k_arr], dtype=np.int32)
    order = np.argsort(pop, kind='stable').astype(np.int32)
    k_by_grade = k_arr[order]

    return {
        'i_lut': torch.tensor(i_lut_list, dtype=torch.int32, device=device).contiguous(),
        'k_offset_i': torch.tensor(k_offset_i, dtype=torch.int32, device=device).contiguous(),
        'sign_lut': torch.from_numpy(sign_arr).to(device).contiguous(),
        'k_offset_sign': torch.tensor(k_offset_sign, dtype=torch.int32, device=device).contiguous(),
        'num_subsets_lut': torch.tensor(num_subsets_lut, dtype=torch.int32, device=device).contiguous(),
        'k_by_grade': torch.from_numpy(k_by_grade).to(device).contiguous(),
    }


def left_contract_subset_grade(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Left contraction. Subset-grade variant — reuses inner_prod_subset_grade_fwd."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = None if metric is None else tuple(int(m) for m in metric)
    L = build_left_contract_subset_lut(n, str(a.device), metric_key)
    return load_inner_prod_subset_grade_cuda().inner_prod_subset_grade_fwd(
        a, b,
        L['i_lut'], L['k_offset_i'],
        L['sign_lut'], L['k_offset_sign'],
        L['num_subsets_lut'], L['k_by_grade'])
