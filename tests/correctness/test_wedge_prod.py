"""Correctness check for the wedge kernels.

Reference: torch_ga `mv_multiply` against the dense outer Cayley tensor
(ShortLex; we permute bp <-> sl to compare).

Both variants of our kernel are checked:
  - chunk-variant `wedge_prod`
  - subset-grade `wedge_prod_subset_grade`

Versor's `wedge` computes (AB - BA)/2 (antisymmetric part of GP), which is
not the standard exterior product on grade >= 2 inputs, so it's not used
here. (It's still in the bench as a timing reference.)
"""

import pytest
import torch

from cliffordcuda.extensions.ga.wedge_prod import (
    wedge_prod, wedge_prod_multik, load_wedge_prod_cuda,
)
from cliffordcuda.extensions.ga.wedge_prod.subset_grade import (
    wedge_prod_subset_grade, load_wedge_prod_subset_grade_cuda,
)
from _cayley import shortlex_to_bp
from _einsum_refs import EinsumWedge

import gacore.kernel as versorai_algebra


def verify_n(n: int, torch_ga, B: int = 2, seed: int = 0):
    from torch_ga.cayley import blades_from_bases, get_cayley_tensor
    from torch_ga.mv_ops import mv_multiply

    dim = 1 << n
    torch.manual_seed(seed)
    a_bp = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    b_bp = torch.randn(B, dim, device='cuda', dtype=torch.float32)

    c_kern    = wedge_prod(a_bp, b_bp)
    c_kern_sg = wedge_prod_subset_grade(a_bp, b_bp)
    c_kern_mk = wedge_prod_multik(a_bp, b_bp)
    diff_sg = float((c_kern - c_kern_sg).abs().max().item())
    diff_mk = float((c_kern - c_kern_mk).abs().max().item())

    sl_to_bp = shortlex_to_bp(n).to('cuda')
    a_sl = a_bp.index_select(-1, sl_to_bp).contiguous()
    b_sl = b_bp.index_select(-1, sl_to_bp).contiguous()

    bases = [chr(ord('a') + i) for i in range(n)]
    blades, _ = blades_from_bases(bases)
    _g, _i, t_outer = get_cayley_tensor([1.0] * n, bases, blades)
    cay_outer = torch.tensor(t_outer, dtype=torch.float32, device='cuda')
    c_sl_ga = mv_multiply(a_sl, b_sl, cay_outer)
    c_sl_kern = c_kern.index_select(-1, sl_to_bp)
    diff_ga = float((c_sl_kern - c_sl_ga).abs().max().item())

    einsum_ref = EinsumWedge(n, device='cuda', dtype=torch.float32)
    diff_einsum = float((c_kern - einsum_ref(a_bp, b_bp)).abs().max().item())
    del einsum_ref

    sig = torch.ones(n, dtype=torch.float32, device='cuda')
    diff_versorai = float((c_kern - versorai_algebra.wedge_product(a_bp, b_bp, sig)).abs().max().item())

    return diff_sg, diff_mk, diff_ga, diff_einsum, diff_versorai


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_wedge_prod(n, torch_ga):
    _ = load_wedge_prod_cuda()
    _ = load_wedge_prod_subset_grade_cuda()
    diff_sg, diff_mk, diff_ga, diff_einsum, diff_versorai = verify_n(n, torch_ga)
    assert diff_sg       < 1e-3, f"|chunk - subset| = {diff_sg}"
    assert diff_mk       < 1e-3, f"|chunk - multik| = {diff_mk}"
    assert diff_ga       < 1e-3, f"|kern - torch_ga| = {diff_ga}"
    assert diff_einsum   < 1e-3, f"|kern - einsum| = {diff_einsum}"
    assert diff_versorai < 1e-3, f"|kern - VersorAI| = {diff_versorai}"
