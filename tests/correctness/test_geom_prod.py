"""Correctness check for the geom_prod CUDA kernel.

Independent references checked at every n:
  (1) torch_ga's `mv_multiply` against the dense geom Cayley (ShortLex,
      tensordot + matmul). Caps near n=10.
  (2) Versor's `CliffordAlgebra.geometric_product` (bit-pattern, (D, D) gather
      + matmul against a precomputed sign table).
  (3) einsum factored reference (EinsumGP).
  (4) VersorAI's `gacore.kernel.geometric_product` with default dispatch
      (bitmasked torch path at n>=7).

Also checks both variants of our kernel:
  - chunk-variant `geom_prod`
  - multi-k-per-warp `geom_prod_multik`
"""

import pytest
import torch

from cliffordcuda.extensions.ga.geom_prod import (
    geom_prod, geom_prod_multik, load_geom_prod_cuda,
)
from _cayley import build_geom_cayley, shortlex_to_bp
from _einsum_refs import EinsumGP

import gacore.kernel as versorai_algebra


def verify_n(n: int, torch_ga_mv_multiply, VersorAlgebra,
             B: int = 2, seed: int = 0):
    dim = 1 << n
    torch.manual_seed(seed)
    a_bp = torch.randn(B, dim, device='cuda')
    b_bp = torch.randn(B, dim, device='cuda')

    # Our two kernel variants (bit-pattern).
    c_kern    = geom_prod(a_bp, b_bp)
    c_kern_mk = geom_prod_multik(a_bp, b_bp)
    diff_mk = float((c_kern - c_kern_mk).abs().max().item())

    # (1) torch_ga ShortLex. Permute bp -> sl once; permute kernel output sl <- bp once.
    sl_to_bp = shortlex_to_bp(n).to('cuda')
    a_sl = a_bp.index_select(-1, sl_to_bp)
    b_sl = b_bp.index_select(-1, sl_to_bp)
    c_kern_sl = c_kern.index_select(-1, sl_to_bp)

    cay_g = build_geom_cayley(n, device='cuda')
    diff_ga = float((torch_ga_mv_multiply(a_sl, b_sl, cay_g) - c_kern_sl).abs().max().item())
    del cay_g, a_sl, b_sl, c_kern_sl

    # (2) Versor (bit-pattern).
    versor = VersorAlgebra(p=n, q=0, r=0, device='cuda')
    diff_versor = float((c_kern - versor.geometric_product(a_bp, b_bp)).abs().max().item())
    del versor

    # (3) einsum reference (bit-pattern, factored outer + sign + XOR scatter).
    einsum_ref = EinsumGP(n, device='cuda', dtype=torch.float32)
    diff_einsum = float((c_kern - einsum_ref(a_bp, b_bp)).abs().max().item())
    del einsum_ref

    # (4) VersorAI (bit-pattern). Default dispatch -> bitmasked torch path at n>=7.
    versorai_sig = torch.ones(n, dtype=torch.float32, device='cuda')
    diff_versorai = float((
        c_kern - versorai_algebra.geometric_product(a_bp, b_bp, versorai_sig)
    ).abs().max().item())

    return diff_mk, diff_ga, diff_versor, diff_einsum, diff_versorai


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9])
def test_geom_prod(n, torch_ga, versor):
    _ = load_geom_prod_cuda()
    from torch_ga.mv_ops import mv_multiply
    diff_mk, diff_ga, diff_versor, diff_einsum, diff_versorai = verify_n(n, mv_multiply, versor)
    assert diff_mk       < 1e-3, f"|chunk - multik| = {diff_mk}"
    assert diff_ga       < 1e-3, f"|kern - torch_ga| = {diff_ga}"
    assert diff_versor   < 1e-3, f"|kern - Versor| = {diff_versor}"
    assert diff_einsum   < 1e-3, f"|kern - einsum| = {diff_einsum}"
    assert diff_versorai < 1e-3, f"|kern - VersorAI| = {diff_versorai}"
