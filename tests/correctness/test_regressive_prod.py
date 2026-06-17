"""Correctness check for the regressive (meet) product.

Three checks per (n, metric):
  (1) chunk variant `regressive_prod`              matches the einsum reference
  (2) subset variant `regressive_prod_subset_grade` matches the same reference
  (3) chunk and subset agree internally (catches differential bugs that both
      could share)

Reference: `EinsumRegressive` from _einsum_refs.py builds the regressive
product from its mathematical definition, `a v b = dual(dual(a) /\\ dual(b))`,
using the metric-aware dual and the einsum exterior product. Independent of
both kernels.

Regressive is undefined for metrics with any degenerate generator (the
pseudoscalar has I.I = 0, so I^{-1} doesn't exist). Tested signatures are
all non-degenerate.
"""
import pytest
import torch

from cliffordcuda.extensions.ga.regressive_prod import (
    regressive_prod, regressive_prod_subset_grade,
)
from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda
from cliffordcuda.extensions.ga.wedge_prod.subset_grade import (
    load_wedge_prod_subset_grade_cuda,
)
from _einsum_refs import EinsumRegressive


SIGNATURES = [
    # n=5: pure +, pure -, Lorentzian, balanced.
    (5, (1, 1, 1, 1, 1)),
    (5, (-1, -1, -1, -1, -1)),
    (5, (1, 1, 1, 1, -1)),
    (5, (1, 1, -1, -1, -1)),
    # n=6: pure +, pure -, balanced.
    (6, (1, 1, 1, 1, 1, 1)),
    (6, (-1, -1, -1, -1, -1, -1)),
    (6, (1, 1, 1, -1, -1, -1)),
    # n=7: pure -, mixed.
    (7, (-1, -1, -1, -1, -1, -1, -1)),
    (7, (1, 1, 1, 1, -1, -1, -1)),
    # n=8: pure -, balanced.
    (8, (-1, -1, -1, -1, -1, -1, -1, -1)),
    (8, (1, 1, 1, 1, -1, -1, -1, -1)),
    # n=9: pure -, mixed-heavy-positive.
    (9, (-1, -1, -1, -1, -1, -1, -1, -1, -1)),
    (9, (1, 1, 1, 1, 1, -1, -1, -1, -1)),
]


@pytest.mark.parametrize("n,metric", SIGNATURES, ids=lambda v: str(v))
def test_regressive_prod(n, metric):
    _ = load_wedge_prod_cuda()
    _ = load_wedge_prod_subset_grade_cuda()
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(2, dim, device='cuda', dtype=torch.float32)

    c_chunk  = regressive_prod(a, b, metric=metric)
    c_subset = regressive_prod_subset_grade(a, b, metric=metric)
    c_ref    = EinsumRegressive(n, metric=metric, device='cuda')(a, b)

    diff_chunk  = float((c_chunk  - c_ref).abs().max().item())
    diff_subset = float((c_subset - c_ref).abs().max().item())
    diff_sg     = float((c_chunk - c_subset).abs().max().item())

    assert diff_chunk  < 1e-3, f"|chunk - einsum| = {diff_chunk}"
    assert diff_subset < 1e-3, f"|subset - einsum| = {diff_subset}"
    assert diff_sg     < 1e-3, f"|chunk - subset| = {diff_sg}"
