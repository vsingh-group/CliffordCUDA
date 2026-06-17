"""Correctness check for the left-contraction kernels.

Reference: Versor's `left_contraction` (bit-pattern, (D, D) gather + matmul
against its `lc_gp_signs` table) — the standard contraction definition.

Both variants of our kernel are checked:
  - chunk variant `left_contract` (kernels/contract.cu)
  - subset-grade variant `left_contract_subset_grade`
"""

import pytest
import torch

from cliffordcuda.extensions.ga.left_contract import (
    left_contract, left_contract_subset_grade, load_contract_cuda,
)
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    load_inner_prod_subset_grade_cuda,
)
from _einsum_refs import EinsumLeftContract

import gacore.kernel as versorai_algebra


def verify_n(n: int, VersorAlgebra, B: int = 2, seed: int = 0):
    dim = 1 << n
    torch.manual_seed(seed)
    a = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(B, dim, device='cuda', dtype=torch.float32)

    c_kern    = left_contract(a, b)
    c_kern_sg = left_contract_subset_grade(a, b)
    diff_sg = float((c_kern - c_kern_sg).abs().max().item())

    valg = VersorAlgebra(p=n, q=0, r=0, device='cuda')
    diff_versor = float((c_kern - valg.left_contraction(a, b)).abs().max().item())
    del valg

    einsum_ref = EinsumLeftContract(n, device='cuda', dtype=torch.float32)
    diff_einsum = float((c_kern - einsum_ref(a, b)).abs().max().item())
    del einsum_ref

    sig = torch.ones(n, dtype=torch.float32, device='cuda')
    diff_versorai = float((c_kern - versorai_algebra.left_contraction(a, b, sig)).abs().max().item())

    return diff_sg, diff_versor, diff_einsum, diff_versorai


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_left_contract(n, versor):
    _ = load_contract_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    diff_sg, diff_versor, diff_einsum, diff_versorai = verify_n(n, versor)
    assert diff_sg       < 1e-3, f"|chunk - subset| = {diff_sg}"
    assert diff_versor   < 1e-3, f"|kern - Versor| = {diff_versor}"
    assert diff_einsum   < 1e-3, f"|kern - einsum| = {diff_einsum}"
    assert diff_versorai < 1e-3, f"|kern - VersorAI| = {diff_versorai}"
