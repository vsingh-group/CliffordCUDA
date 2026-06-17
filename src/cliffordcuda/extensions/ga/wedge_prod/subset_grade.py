"""Python wrapper for wedge subset-iter + grade-ordered warps (Idea 1 + 2).

LUTs are built once per (n, device) and cached:
  - i_lut[k_offset_i[k] : k_offset_i[k] + 2^|k|] = subsets of k in lex-of-t order
  - sign_lut[k_offset_sign[k] : k_offset_sign[k] + ceil(2^|k|/32)] is packed
    32 sigma bits per int32, matching the kernel's (warp_iter, lane) access.
  - k_by_grade is a permutation of [0, dim) sorted by popcount(k), so all
    warps in a block share the same |k|.
"""
import functools
import os

import numpy as np
import torch

from ...._config import _GA_KERNELS_DIR, load_extension


def load_wedge_prod_subset_grade_cuda():
    if not hasattr(load_wedge_prod_subset_grade_cuda, '_module'):
        load_wedge_prod_subset_grade_cuda._module = load_extension(
            name='wedge_prod_subset_grade_cuda',
            sources=[os.path.join(_GA_KERNELS_DIR, 'wedge_prod_subset_grade.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return load_wedge_prod_subset_grade_cuda._module


def _sigma(i: int, j: int) -> int:
    s = 0
    ii = i >> 1
    while ii:
        s ^= bin(ii & j).count('1') & 1
        ii >>= 1
    return -1 if (s & 1) else 1


@functools.lru_cache(maxsize=None)
def build_wedge_subset_lut(n: int, device: str = 'cuda'):
    if n < 5:
        raise ValueError("n>=5 required")
    dim = 1 << n
    i_lut_list = []
    sign_lut_list = []
    k_offset_i = [0] * dim
    k_offset_sign = [0] * dim
    cum_i = 0
    cum_sign = 0

    for k in range(dim):
        k_offset_i[k] = cum_i
        k_offset_sign[k] = cum_sign

        bit_positions = [b for b in range(n) if k & (1 << b)]
        p = len(bit_positions)
        n_subsets = 1 << p
        n_iters = (n_subsets + 31) // 32

        i_vals = []
        for t in range(n_subsets):
            i_val = 0
            for b_idx, b_pos in enumerate(bit_positions):
                if (t >> b_idx) & 1:
                    i_val |= (1 << b_pos)
            i_vals.append(i_val)
        i_lut_list.extend(i_vals)

        for w in range(n_iters):
            word = 0
            for lane in range(32):
                t_global = w * 32 + lane
                if t_global >= n_subsets:
                    continue
                i_val = i_vals[t_global]
                j_val = k ^ i_val
                if _sigma(i_val, j_val) == -1:
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
        'k_by_grade': torch.from_numpy(k_by_grade).to(device).contiguous(),
    }


def wedge_prod_subset_grade(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Idea 1 + 2: subset iteration with grade-ordered warps."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    luts = build_wedge_subset_lut(n, str(a.device))
    return load_wedge_prod_subset_grade_cuda().wedge_prod_subset_grade_fwd(
        a, b,
        luts['i_lut'], luts['k_offset_i'],
        luts['sign_lut'], luts['k_offset_sign'],
        luts['k_by_grade'])
