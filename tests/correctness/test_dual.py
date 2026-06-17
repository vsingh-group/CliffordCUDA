"""Correctness check for the dual `x* = x . I^{-1}`.

The dual is implemented as a signed reversal of the bit-pattern coefficients
(`x.flip(-1) * signs`, see `CliffordAlgebra.dual`). This checks it against its
own definition run through the geometric-product kernel:

    dual(x) == geom_prod(x, I^{-1})

`geom_prod` is validated independently against the witness libraries in
test_geom_prod and shares no code with the flip+sign path, so this confirms
both the signs and that the bit-pattern `flip` realises the right complement
permutation.

Both the eager and `compile=True` paths are checked.

Needs an invertible pseudoscalar (non-degenerate metric) and the geom_prod
kernel (n >= 5).
"""
import pytest
import torch

from cliffordcuda import CliffordAlgebra


SIGNATURES = [
    # n=5: pure +, Lorentzian, balanced.
    (5, (1, 1, 1, 1, 1)),
    (5, (1, 1, 1, 1, -1)),
    (5, (1, 1, -1, -1, -1)),
    # n=6: pure -, balanced.
    (6, (-1, -1, -1, -1, -1, -1)),
    (6, (1, 1, 1, -1, -1, -1)),
    # n=7, 8, 9: mixed.
    (7, (1, 1, 1, 1, -1, -1, -1)),
    (8, (-1, -1, -1, -1, -1, -1, -1, -1)),
    (9, (1, 1, 1, 1, 1, -1, -1, -1, -1)),
]


@pytest.mark.parametrize("n,metric", SIGNATURES, ids=lambda v: str(v))
def test_dual(n, metric):
    dim = 1 << n
    full = dim - 1
    torch.manual_seed(0)
    x = torch.randn(2, dim, device='cuda', dtype=torch.float32)

    cl = CliffordAlgebra(metric=list(metric), device='cuda')

    # Reference: dual(x) = x . I^{-1} via the geom_prod kernel.
    # I^{-1} = (1 / I^2) e_full, with I^2 = (-1)^{n(n-1)/2} * prod(metric).
    i_sq = (-1) ** (n * (n - 1) // 2)
    for m in metric:
        i_sq *= m
    I_inv = torch.zeros(2, dim, device='cuda', dtype=torch.float32)
    I_inv[:, full] = 1.0 / i_sq
    ref = cl.geom_prod(x, I_inv)

    d_eager = cl.dual(x)
    d_comp  = cl.dual(x, compile=True)

    diff_eager = float((d_eager - ref).abs().max().item())
    diff_comp  = float((d_comp - d_eager).abs().max().item())

    assert diff_eager < 1e-5, f"|dual - geom_prod(x, I^-1)| = {diff_eager}"
    assert diff_comp  < 1e-5, f"|dual(compile=True) - dual| = {diff_comp}"
