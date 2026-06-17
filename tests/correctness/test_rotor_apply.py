"""Correctness check for rotor application — chunk vs Versor vs einsum.

For a shared random bivector (lex-pair order, C(n, 2) coefficients) and a
shared random multivector x:

  chunk     CliffordAlgebra.compile_bivector + apply_rotor (bit-pattern x).
  Versor    VersorAlgebra.exp(bivector_mv) + per_channel_sandwich.
  einsum    two EinsumGP calls implementing R~ x R, where R comes from
            Versor's exp (independent forward path; uses metric-aware
            sigma * disjoint-mask scatter, not a fused kernel).

Cl(n, 0) ONLY. The chunk rotor pipeline (`compile_bivector` -> `EighExpFunc`
-> `apply_rotor`) does not take the metric; the kernel is hardcoded for the
Euclidean signature and the rotation tables it precomputes don't account
for negative or degenerate generators. Adding non-Euclidean test cases would
exercise that limitation, not measure correctness on supported inputs.

torch_ga is intentionally NOT checked here. The bench-side TorchGARotor
draws R from a random even-grade multivector rather than building it from
the shared bivector via exp (upstream `exp` only accepts inputs that
square to a scalar; `approx_exp` overflows fp32 at n >= 9; CPU fp64 is
too slow). Comparing chunk's rotor to a random R isn't meaningful, so
torch_ga lives only in the timing / memory benches.
"""

import pytest
import torch

from cliffordcuda import CliffordAlgebra
from _cayley import shortlex_to_bp
from _einsum_refs import EinsumGP
from _rotor_apply_helpers import VersorRotor

import gacore.kernel as versorai_algebra


def verify_n(n: int, B: int = 2, seed: int = 0):
    dim = 1 << n
    torch.manual_seed(seed)

    num_basis_biv = n * (n - 1) // 2
    biv = torch.randn(1, num_basis_biv, device='cuda')
    x_bp = torch.randn(B, dim, device='cuda')

    # 1. chunk via the public API (bit-pattern input/output).
    cl = CliffordAlgebra(metric=[1] * n, device='cuda')
    cs = cl.compile_bivector(biv)
    with torch.no_grad():
        y_chunk_bp = cl.apply_rotor(cs, x_bp)
    del cl, cs

    # 2. Versor (bit-pattern x). Also extract R and R~ so the einsum path
    #    evaluates the same sandwich (R~ x R) independently.
    versor = VersorRotor(dim=dim, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        versor.bivectors_left.copy_(biv)
    versor._update_rotors()
    with torch.no_grad():
        y_versor_bp = versor(x_bp)
        R     = versor._rotor.detach().clone()
        R_rev = versor._rotor_rev.detach().clone()
    del versor

    # 3. einsum: two EinsumGP calls on Versor's R and R~.
    einsum_gp = EinsumGP(n, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        y_einsum_bp = einsum_gp(einsum_gp(R_rev, x_bp), R)
    del einsum_gp

    # 4. VersorAI: two `geometric_product` calls on Versor's R and R~.
    sig = torch.ones(n, dtype=torch.float32, device='cuda')
    vgp = versorai_algebra.geometric_product
    with torch.no_grad():
        y_versorai_bp = vgp(vgp(R_rev, x_bp, sig), R, sig)

    diff_versor   = float((y_chunk_bp - y_versor_bp).abs().max().item())
    diff_einsum   = float((y_chunk_bp - y_einsum_bp).abs().max().item())
    diff_versorai = float((y_chunk_bp - y_versorai_bp).abs().max().item())
    return diff_versor, diff_einsum, diff_versorai


@pytest.mark.parametrize("n", [7, 8, 9, 10])
def test_rotor_apply(n, versor):
    diff_v, diff_e, diff_vai = verify_n(n)
    dim = 1 << n
    # Versor uses a single (D, D) per_channel_sandwich matmul that loses
    # more fp32 precision at unit-magnitude bivectors (rotor mv has terms
    # that cancel in the matmul); allow a looser tolerance.
    tol = max(1e-1, dim * 1e-3)
    assert diff_v   < tol, f"|chunk - Versor| = {diff_v} >= {tol}"
    assert diff_e   < tol, f"|chunk - einsum| = {diff_e} >= {tol}"
    assert diff_vai < tol, f"|chunk - VersorAI| = {diff_vai} >= {tol}"
