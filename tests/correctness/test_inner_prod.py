"""Correctness check for the inner (Hestenes) kernels.

Reference: torch_ga `mv_multiply` against the dense inner Cayley tensor
(ShortLex).

Both variants of our kernel are checked:
  - chunk-variant `inner_prod`
  - subset-grade `inner_prod_subset_grade`

Versor's `inner_product` is (AB + BA)/2 (symmetric part of GP), not the
Hestenes inner on grade >= 2 inputs, so it's not used here.
"""

import pytest
import torch

from cliffordcuda.extensions.ga.inner_prod import (
    inner_prod, inner_prod_multik, load_inner_prod_cuda,
)
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    inner_prod_subset_grade, load_inner_prod_subset_grade_cuda,
)
from _cayley import shortlex_to_bp
from _einsum_refs import EinsumInner



def _build_inner_cayley_via_torch_ga(n: int) -> torch.Tensor:
    from torch_ga.cayley import blades_from_bases, get_cayley_tensor
    bases = [chr(ord('a') + i) for i in range(n)]
    blades, _ = blades_from_bases(bases)
    _, t_inner, _ = get_cayley_tensor([1.0] * n, bases, blades)
    return torch.tensor(t_inner, dtype=torch.float32, device='cuda')


def verify_n(n: int, B: int = 2, seed: int = 0):
    from torch_ga.mv_ops import mv_multiply

    dim = 1 << n
    torch.manual_seed(seed)
    a_bp = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    b_bp = torch.randn(B, dim, device='cuda', dtype=torch.float32)

    c_kern    = inner_prod(a_bp, b_bp)
    c_kern_sg = inner_prod_subset_grade(a_bp, b_bp)
    c_kern_mk = inner_prod_multik(a_bp, b_bp)
    diff_sg = float((c_kern - c_kern_sg).abs().max().item())
    diff_mk = float((c_kern - c_kern_mk).abs().max().item())

    sl_to_bp = shortlex_to_bp(n).to('cuda')
    a_sl = a_bp.index_select(-1, sl_to_bp).contiguous()
    b_sl = b_bp.index_select(-1, sl_to_bp).contiguous()
    cay = _build_inner_cayley_via_torch_ga(n)
    c_sl_ga = mv_multiply(a_sl, b_sl, cay)
    c_sl_kern = c_kern.index_select(-1, sl_to_bp)
    diff_ga = float((c_sl_kern - c_sl_ga).abs().max().item())

    einsum_ref = EinsumInner(n, device='cuda', dtype=torch.float32)
    diff_einsum = float((c_kern - einsum_ref(a_bp, b_bp)).abs().max().item())
    del einsum_ref
    return diff_sg, diff_mk, diff_ga, diff_einsum


# VersorAI's `inner_product` is filtered_product(mode="inner") which adds
# `mask * (gi > 0) * (gj > 0)` — i.e. it zeros out the result when either
# input has any grade-0 component, matching the "fat dot" / Lounesto
# scalar-exclusion convention. That's a different operation from Hestenes
# on randn inputs, so VersorAI is in the inner_prod bench as a timing
# reference but not asserted here.


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_inner_prod(n, torch_ga):
    _ = load_inner_prod_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    diff_sg, diff_mk, diff_ga, diff_einsum = verify_n(n)
    assert diff_sg     < 1e-3, f"|chunk - subset| = {diff_sg}"
    assert diff_mk     < 1e-3, f"|chunk - multik| = {diff_mk}"
    assert diff_ga     < 1e-3, f"|kern - torch_ga| = {diff_ga}"
    assert diff_einsum < 1e-3, f"|kern - einsum| = {diff_einsum}"
