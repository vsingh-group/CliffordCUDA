"""Correctness check for the right-contraction kernels.

References:
  - chunk vs subset (internal agreement between our two variants)
  - einsum: factored outer + sign + XOR scatter (independent PyTorch impl)

No general right contraction in Versor (only a bivector x vector
specialization), so Versor isn't used here.
"""
import pytest
import torch

from cliffordcuda.extensions.ga.right_contract import (
    right_contract, right_contract_subset_grade, load_contract_cuda,
)
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    load_inner_prod_subset_grade_cuda,
)
from _einsum_refs import EinsumRightContract


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_right_contract(n):
    _ = load_contract_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(2, dim, device='cuda', dtype=torch.float32)

    c_chunk  = right_contract(a, b)
    c_subset = right_contract_subset_grade(a, b)
    diff_sg = float((c_chunk - c_subset).abs().max().item())

    einsum_ref = EinsumRightContract(n, device='cuda', dtype=torch.float32)
    diff_einsum = float((c_chunk - einsum_ref(a, b)).abs().max().item())

    assert diff_sg     < 1e-3, f"|chunk - subset| = {diff_sg}"
    assert diff_einsum < 1e-3, f"|kern - einsum| = {diff_einsum}"
