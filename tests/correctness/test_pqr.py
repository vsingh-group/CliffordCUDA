"""Correctness check for Cl(p, q, r) generalization across all kernels.

Verifies geom_prod, inner_prod, wedge_prod and their subset-grade variants
against torch_ga's dense Cayley tensor for several signatures:
  - Cl(5, 0): Euclidean (sanity, fast path)
  - Cl(0, 5): all negative (non-degenerate)
  - Cl(4, 1): Lorentzian / CGA-like (non-degenerate)
  - Cl(2, 3): mixed
  - Cl(4, 0, 1): one degenerate generator (PGA-like)
  - Cl(3, 1, 1): mixed with one degenerate generator
  - Cl(5, 0, 1) at n=6: a degenerate generator added to Euclidean
  - Cl(3, 2, 1) at n=6

torch_ga's Cayley tensor is in ShortLex (grade-lex) blade ordering; we permute
multivectors between bit-pattern and ShortLex to compare. Versor is used as a
second witness for the geometric product only.
"""
import pytest
import torch

from cliffordcuda.extensions.ga.geom_prod import (
    geom_prod, geom_prod_multik, load_geom_prod_cuda,
    _build_sign_value_table, _normalize_metric,
)
from cliffordcuda.extensions.ga.inner_prod import inner_prod, load_inner_prod_cuda
from cliffordcuda.extensions.ga.wedge_prod import wedge_prod, load_wedge_prod_cuda
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    inner_prod_subset_grade, load_inner_prod_subset_grade_cuda,
)
from cliffordcuda.extensions.ga.wedge_prod.subset_grade import (
    wedge_prod_subset_grade, load_wedge_prod_subset_grade_cuda,
)
from cliffordcuda.extensions.ga.left_contract import (
    left_contract, left_contract_subset_grade,
)
from cliffordcuda.extensions.ga.right_contract import (
    right_contract, right_contract_subset_grade,
)
from _cayley import shortlex_to_bp


def _reference_contract(a: torch.Tensor, b: torch.Tensor, metric, direction: str) -> torch.Tensor:
    """Bit-pattern reference for left/right contraction: outer * masked sigma
    table -> XOR-scatter-add. Same blade indexing as the kernels."""
    n = len(metric)
    metric = _normalize_metric(n, metric)
    dim = 1 << n
    sv = _build_sign_value_table(n, metric).to(torch.float32).to(a.device)
    i_grid = torch.arange(dim, device=a.device).view(-1, 1).expand(dim, dim)
    j_grid = torch.arange(dim, device=a.device).view(1, -1).expand(dim, dim)
    if direction == 'left':                # i is subset of j
        mask = ((i_grid & j_grid) == i_grid).to(torch.float32)
    elif direction == 'right':             # j is subset of i
        mask = ((i_grid & j_grid) == j_grid).to(torch.float32)
    else:
        raise ValueError(direction)
    weight = sv * mask
    outer = a.unsqueeze(-1) * b.unsqueeze(-2)
    weighted = outer * weight
    k_idx = (i_grid ^ j_grid).to(torch.long)
    flat_w = weighted.reshape(*weighted.shape[:-2], -1)
    flat_k = k_idx.reshape(-1).expand_as(flat_w)
    out = torch.zeros_like(a)
    return out.scatter_add(-1, flat_k, flat_w)


def _build_cayleys(n: int, metric):
    from torch_ga.cayley import blades_from_bases, get_cayley_tensor
    bases = [chr(ord('a') + i) for i in range(n)]
    blades, _ = blades_from_bases(bases)
    metric_f = [float(m) for m in metric]
    t_geom, t_inner, t_outer = get_cayley_tensor(metric_f, bases, blades)
    dev = 'cuda'
    return (
        torch.tensor(t_geom,  dtype=torch.float32, device=dev),
        torch.tensor(t_inner, dtype=torch.float32, device=dev),
        torch.tensor(t_outer, dtype=torch.float32, device=dev),
    )


def verify_signature(n: int, metric, VersorAlgebra,
                     B: int = 2, seed: int = 0, tol: float = 1e-3):
    from torch_ga.mv_ops import mv_multiply

    dim = 1 << n
    torch.manual_seed(seed)
    a_bp = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    b_bp = torch.randn(B, dim, device='cuda', dtype=torch.float32)

    sl_to_bp = shortlex_to_bp(n).to('cuda')
    a_sl = a_bp.index_select(-1, sl_to_bp).contiguous()
    b_sl = b_bp.index_select(-1, sl_to_bp).contiguous()

    t_g, t_i, t_o = _build_cayleys(n, metric)

    pos = sum(1 for m in metric if m == 1)
    neg = sum(1 for m in metric if m == -1)
    deg = sum(1 for m in metric if m == 0)
    valg = VersorAlgebra(p=pos, q=neg, r=deg, device='cuda')

    def _diff_perm(c_kern_bp, c_ref_sl):
        return float((c_kern_bp.index_select(-1, sl_to_bp) - c_ref_sl).abs().max().item())

    diffs = {}
    # geom_prod (chunk + multik)
    c_kern_g  = geom_prod(a_bp, b_bp, metric=metric)
    c_kern_gm = geom_prod_multik(a_bp, b_bp, metric=metric)
    c_ref_g   = mv_multiply(a_sl, b_sl, t_g)
    diffs["GP"]   = _diff_perm(c_kern_g,  c_ref_g)
    diffs["GPmk"] = _diff_perm(c_kern_gm, c_ref_g)
    diffs["GPversor"] = float((c_kern_g - valg.geometric_product(a_bp, b_bp)).abs().max().item())

    # inner_prod (chunk + subset_grade)
    c_kern_i   = inner_prod(a_bp, b_bp, metric=metric)
    c_kern_isg = inner_prod_subset_grade(a_bp, b_bp, metric=metric)
    c_ref_i    = mv_multiply(a_sl, b_sl, t_i)
    diffs["IP"]   = _diff_perm(c_kern_i,   c_ref_i)
    diffs["IPsg"] = _diff_perm(c_kern_isg, c_ref_i)

    # wedge_prod (signature-independent at the term level)
    c_kern_w   = wedge_prod(a_bp, b_bp)
    c_kern_wsg = wedge_prod_subset_grade(a_bp, b_bp)
    c_ref_w    = mv_multiply(a_sl, b_sl, t_o)
    diffs["WP"]   = _diff_perm(c_kern_w,   c_ref_w)
    diffs["WPsg"] = _diff_perm(c_kern_wsg, c_ref_w)

    # Left + right contraction
    c_ref_l = _reference_contract(a_bp, b_bp, metric, direction='left')
    diffs["L"]   = float((left_contract(a_bp, b_bp, metric=metric)              - c_ref_l).abs().max().item())
    diffs["Lsg"] = float((left_contract_subset_grade(a_bp, b_bp, metric=metric) - c_ref_l).abs().max().item())
    c_ref_r = _reference_contract(a_bp, b_bp, metric, direction='right')
    diffs["R"]   = float((right_contract(a_bp, b_bp, metric=metric)              - c_ref_r).abs().max().item())
    diffs["Rsg"] = float((right_contract_subset_grade(a_bp, b_bp, metric=metric) - c_ref_r).abs().max().item())

    # einsum references for each product under this metric.
    from _einsum_refs import (
        EinsumGP, EinsumWedge, EinsumInner,
        EinsumLeftContract, EinsumRightContract,
    )
    diffs["GPeinsum"] = float((c_kern_g - EinsumGP(n, metric=metric, device='cuda')(a_bp, b_bp)).abs().max().item())
    diffs["WPeinsum"] = float((c_kern_w - EinsumWedge(n, metric=metric, device='cuda')(a_bp, b_bp)).abs().max().item())
    diffs["IPeinsum"] = float((c_kern_i - EinsumInner(n, metric=metric, device='cuda')(a_bp, b_bp)).abs().max().item())
    diffs["Leinsum"]  = float((left_contract(a_bp, b_bp, metric=metric)  - EinsumLeftContract(n, metric=metric, device='cuda')(a_bp, b_bp)).abs().max().item())
    diffs["Reinsum"]  = float((right_contract(a_bp, b_bp, metric=metric) - EinsumRightContract(n, metric=metric, device='cuda')(a_bp, b_bp)).abs().max().item())

    return diffs


SIGNATURES = [
    # n=5: full spread — pure +, pure -, Lorentzian, balanced, degenerate, mixed.
    (5, (1, 1, 1, 1, 1)),
    (5, (-1, -1, -1, -1, -1)),
    (5, (1, 1, 1, 1, -1)),
    (5, (1, 1, -1, -1, -1)),
    (5, (1, 1, 1, 1, 0)),
    (5, (1, 1, 1, -1, 0)),
    # n=6: pure-degenerate and mixed-degenerate.
    (6, (1, 1, 1, 1, 1, 0)),
    (6, (1, 1, 1, -1, -1, 0)),
    # n=7: pure +, Lorentzian-like, mixed + degenerate.
    (7, (1, 1, 1, 1, 1, 1, 1)),
    (7, (1, 1, 1, 1, 1, 1, -1)),
    (7, (1, 1, 1, 1, -1, -1, 0)),
    # n=8: mixed signature, mixed + degenerate.
    (8, (1, 1, 1, 1, 1, 1, 1, -1)),
    (8, (1, 1, 1, 1, -1, -1, -1, 0)),
    # n=9: Lorentzian-like, mixed + degenerate.
    (9, (1, 1, 1, 1, 1, 1, 1, 1, -1)),
    (9, (1, 1, 1, 1, 1, -1, -1, -1, 0)),
]


@pytest.mark.parametrize("n,metric", SIGNATURES, ids=lambda v: str(v))
def test_pqr(n, metric, torch_ga, versor):
    _ = load_geom_prod_cuda()
    _ = load_inner_prod_cuda()
    _ = load_wedge_prod_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    _ = load_wedge_prod_subset_grade_cuda()
    diffs = verify_signature(n, metric, versor)
    failed = {k: v for k, v in diffs.items() if v >= 1e-3}
    assert not failed, f"Cl{metric} failed: {failed}"
