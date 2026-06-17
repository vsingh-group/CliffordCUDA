"""Python wrapper for inner subset-iter + grade-ordered warps (Idea 1 + 2).

Generalized to Cl(p, q, r): for degenerate metrics (r > 0) the LUT filters out
(i, j) pairs whose shared bases hit a generator with metric 0. The sign LUT
encodes the full sigma_val (reorder parity * metric factor product), which is
±1 for surviving terms. The kernel itself is unchanged — it just iterates a
shorter num_subsets_lut[k] when zero-valued terms have been filtered out.
"""
import functools
import os

import numpy as np
import torch

from ...._config import _GA_KERNELS_DIR, load_extension
from ..geom_prod import _normalize_metric


def load_inner_prod_subset_grade_cuda():
    if not hasattr(load_inner_prod_subset_grade_cuda, '_module'):
        load_inner_prod_subset_grade_cuda._module = load_extension(
            name='inner_prod_subset_grade_cuda',
            sources=[os.path.join(_GA_KERNELS_DIR, 'inner_prod_subset_grade.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_inner_prod_subset_grade_cuda._module


def _sigma_val(i: int, j: int, metric) -> int:
    """Full sigma_val(i, j) in {-1, 0, +1} for the given metric (length-n tuple)."""
    s = 0
    ii = i >> 1
    while ii:
        s ^= bin(ii & j).count('1') & 1
        ii >>= 1
    sign = -1 if (s & 1) else 1
    common = i & j
    factor = 1
    for k, m in enumerate(metric):
        if (common >> k) & 1:
            if m == 0:
                return 0
            if m == -1:
                factor = -factor
    return sign * factor


@functools.lru_cache(maxsize=None)
def build_inner_subset_lut(n: int, device: str = 'cuda', metric=None):
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

        # Enumerate x ⊆ comp; shared bases of (i, j) equal x in both case1/case2.
        # Drop x that hits any degenerate base.
        x_vals = []
        for t in range(n_x):
            x = 0
            for b_idx, b_pos in enumerate(comp_bits):
                if (t >> b_idx) & 1:
                    x |= (1 << b_pos)
            if (x & deg_mask) != 0:
                continue
            x_vals.append(x)

        if k == 0:
            i_vals = x_vals
        else:
            case1 = [(k | x) for x in x_vals]
            case2 = list(x_vals)
            i_vals = case1 + case2

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
                # By construction surviving terms have sv != 0.
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


def inner_prod_subset_grade(a: torch.Tensor, b: torch.Tensor, metric=None) -> torch.Tensor:
    """Idea 1 + 2: subset iteration with grade-ordered warps. metric=None -> Cl(n, 0)."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_key = None if metric is None else tuple(int(m) for m in metric)
    L = build_inner_subset_lut(n, str(a.device), metric_key)
    return load_inner_prod_subset_grade_cuda().inner_prod_subset_grade_fwd(
        a, b,
        L['i_lut'], L['k_offset_i'],
        L['sign_lut'], L['k_offset_sign'],
        L['num_subsets_lut'], L['k_by_grade'])
