"""Correctness check for the squared norm `norm_sq(x) = <x x~>_0`.

norm_sq is implemented as a metric-weighted reduction `sum_b w_b * x_b^2`
(see `CliffordAlgebra.norm_sq`). This checks it against the definition run
through the geom_prod kernel and reverse: `<x x~>_0` is the scalar part
(grade 0 = bit-pattern index 0) of `geom_prod(x, reverse(x))`. geom_prod is
validated independently in test_geom_prod and shares no code with the
weighted-reduction path.

Both the eager and `compile=True` paths are checked. norm_sq is defined for
every signature, so Euclidean, indefinite, and degenerate metrics are all
exercised.
"""
import pytest
import torch

from cliffordcuda import CliffordAlgebra


# Covers every combination of p>0 / q>0 / r>0 the op supports (norm_sq is
# defined for all of them).
SIGNATURES = [
    # Euclidean and purely indefinite (r = 0).
    (5, (1, 1, 1, 1, 1)),            # Cl(5, 0, 0)
    (5, (1, 1, 1, 1, -1)),           # Cl(4, 1, 0)
    (5, (1, 1, -1, -1, -1)),         # Cl(2, 3, 0)
    (6, (-1, -1, -1, -1, -1, -1)),   # Cl(0, 6, 0)
    (6, (1, 1, 1, -1, -1, -1)),      # Cl(3, 3, 0)
    (7, (1, 1, 1, 1, -1, -1, -1)),   # Cl(4, 3, 0)
    (8, (-1, -1, -1, -1, -1, -1, -1, -1)),     # Cl(0, 8, 0)
    (9, (1, 1, 1, 1, 1, -1, -1, -1, -1)),      # Cl(5, 4, 0)
    # Degenerate, q = 0 (only + and 0).
    (5, (1, 1, 1, 1, 0)),            # Cl(4, 0, 1)
    (6, (1, 1, 1, 1, 1, 0)),         # Cl(5, 0, 1)
    # Degenerate, p = 0 (only - and 0).
    (5, (-1, -1, -1, -1, 0)),        # Cl(0, 4, 1)
    # Fully mixed: p, q, r all > 0.
    (5, (1, 1, -1, -1, 0)),          # Cl(2, 2, 1)
    (6, (1, 1, 1, -1, -1, 0)),       # Cl(3, 2, 1)
    (6, (1, 1, -1, -1, 0, 0)),       # Cl(2, 2, 2)
    (7, (1, 1, 1, -1, -1, -1, 0)),   # Cl(3, 3, 1)
]


@pytest.mark.parametrize("n,metric", SIGNATURES, ids=lambda v: str(v))
def test_norm_sq(n, metric):
    dim = 1 << n
    torch.manual_seed(0)
    x = torch.randn(2, dim, device='cuda', dtype=torch.float32)

    cl = CliffordAlgebra(metric=list(metric), device='cuda')

    # Reference: <x x~>_0 = scalar (grade 0 = bit-pattern index 0) part of
    # geom_prod(x, reverse(x)).
    ref = cl.geom_prod(x, cl.reverse(x))[:, 0]

    ns_eager = cl.norm_sq(x)
    ns_comp  = cl.norm_sq(x, compile=True)

    assert torch.allclose(ns_eager, ref, rtol=1e-4, atol=1e-3), \
        f"|norm_sq - <x x~>_0| max = {float((ns_eager - ref).abs().max())}"
    assert torch.allclose(ns_comp, ns_eager, rtol=1e-5, atol=1e-4), \
        f"|norm_sq(compile=True) - norm_sq| max = {float((ns_comp - ns_eager).abs().max())}"
