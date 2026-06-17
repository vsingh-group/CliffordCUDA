"""Numerical gradient check for geom_prod's autograd backward.

Compares the analytic backward (two more geom_prod_fwd kernel calls with the
direct-sigma table) against PyTorch's finite-difference numerical gradient,
across the six Cl(p, q, r) signature shapes per n. See `tests/_gradcheck.py`
for the shared signature spread + wrapper.
"""
import pytest

from cliffordcuda.extensions.ga.geom_prod import geom_prod, load_geom_prod_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_geom_prod(n, metric):
    _ = load_geom_prod_cuda()
    assert run_gradcheck(geom_prod, n, metric)
