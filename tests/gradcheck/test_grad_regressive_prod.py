"""Gradcheck for regressive_prod.

Restricted to non-degenerate signatures: regressive is undefined for r > 0
(the pseudoscalar has I . I = 0, no inverse).
"""
import pytest

from cliffordcuda.extensions.ga.regressive_prod import regressive_prod
from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda
from cliffordcuda.extensions.ga.geom_prod import load_geom_prod_cuda
from _gradcheck import run_gradcheck, N_VALUES_FAST


def _non_degenerate_cases():
    """Same shape as `_gradcheck.all_cases` but only the first four signatures
    per n (drop the two with r > 0)."""
    out = []
    for n in N_VALUES_FAST:
        out.extend([
            (n, tuple([1] * n)),
            (n, tuple([-1] * n)),
            (n, tuple([1] * (n - 1) + [-1])),
            (n, tuple([1] * (n // 2) + [-1] * (n - n // 2))),
        ])
    return out


@pytest.mark.parametrize("n,metric", _non_degenerate_cases())
def test_grad_regressive_prod(n, metric):
    _ = load_wedge_prod_cuda()
    _ = load_geom_prod_cuda()
    assert run_gradcheck(regressive_prod, n, metric)
