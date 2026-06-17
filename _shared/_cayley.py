"""Dense Cayley tensors in grade-lex (ShortLex) blade ordering, vectorized.

torch_ga's `get_cayley_tensor` is a Python double loop with O(dim) blade
lookups, so its O(dim^3) total work dominates any benchmark at n >= 9. These
functions build the same tensors with bit-pattern algebra and torch ops; they
match torch_ga's output element-for-element but are ~1000x faster at n=10.

  shortlex_to_bp(n)     — permutation: ShortLex index -> bit-pattern index.
  build_geom_cayley(n)  — full geometric product, Cl(n, 0).
  build_inner_cayley(n) — Hestenes inner (drops to grade |i|-|j|).
  build_outer_cayley(n) — exterior / wedge product.

All return (dim, dim, dim) float32 tensors on CUDA in torch_ga's
convention — `cay[left, right, output]` — so that
`mv_multiply(a, b, cay)` computes `out[k] = Σ a[i] * cay[i, j, k] * b[j]`.
"""
from itertools import combinations

import torch


def shortlex_to_bp(n: int) -> torch.Tensor:
    """`out[k]` = bit-pattern index of the k-th ShortLex (grade-lex) blade."""
    out = []
    for grade in range(n + 1):
        for tup in combinations(range(n), grade):
            out.append(sum(1 << g for g in tup))
    return torch.tensor(out, dtype=torch.long)


def _sigma_table(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor]:
    """Returns (sign(dim, dim), bm_i, bm_j, bp_to_sl) for Cl(n, 0).
    `sign[i, j]` = reorder parity of e_i * e_j in bit-pattern indexing."""
    dim = 1 << n
    sl_to_bp = shortlex_to_bp(n)
    bp_to_sl = torch.empty(dim, dtype=torch.int64)
    bp_to_sl[sl_to_bp] = torch.arange(dim, dtype=torch.int64)

    bm_i = sl_to_bp.view(dim, 1).expand(dim, dim)
    bm_j = sl_to_bp.view(1, dim).expand(dim, dim)
    parity = torch.zeros(dim, dim, dtype=torch.int64)
    for a in range(n):
        bit_a = (bm_i >> a) & 1
        for b in range(a):
            bit_b = (bm_j >> b) & 1
            parity += bit_a * bit_b
    sign = (1 - 2 * (parity & 1)).to(torch.float32)
    return sign, bm_i, bm_j, bp_to_sl


def _build_cayley(n: int, mask: torch.Tensor, v_op: str,
                  device: str = "cuda") -> torch.Tensor:
    """Common driver. mask: (dim, dim) float32 0/1 op-validity. v_op in
    {'xor', 'or'} picks how the output blade is built from i and j.

    Convention: cay[left, right, output] — matches torch_ga / get_cayley_tensor."""
    dim = 1 << n
    sign, bm_i, bm_j, bp_to_sl = _sigma_table(n)
    bm_v = (bm_i ^ bm_j) if v_op == "xor" else (bm_i | bm_j)
    v_sl = bp_to_sl[bm_v]

    cay = torch.zeros(dim, dim, dim, dtype=torch.float32)
    i_sl = torch.arange(dim, dtype=torch.int64).view(dim, 1).expand(dim, dim)
    j_sl = torch.arange(dim, dtype=torch.int64).view(1, dim).expand(dim, dim)
    # cay[left=i, right=j, output=v] = sign * mask
    cay[i_sl, j_sl, v_sl] = sign * mask
    return cay.to(device=device)


def build_geom_cayley(n: int, device: str = "cuda") -> torch.Tensor:
    """Full GP Cayley tensor for Cl(n, 0)."""
    dim = 1 << n
    mask = torch.ones(dim, dim, dtype=torch.float32)
    return _build_cayley(n, mask, "xor", device)


def build_inner_cayley(n: int, device: str = "cuda") -> torch.Tensor:
    """Hestenes inner: (i ⊆ j) OR (j ⊆ i)."""
    _, bm_i, bm_j, _ = _sigma_table(n)
    mask = (((bm_i & (~bm_j)) == 0) | ((bm_j & (~bm_i)) == 0)).to(torch.float32)
    return _build_cayley(n, mask, "xor", device)


def build_outer_cayley(n: int, device: str = "cuda") -> torch.Tensor:
    """Wedge / exterior: (i & j) == 0."""
    _, bm_i, bm_j, _ = _sigma_table(n)
    mask = ((bm_i & bm_j) == 0).to(torch.float32)
    return _build_cayley(n, mask, "or", device)
