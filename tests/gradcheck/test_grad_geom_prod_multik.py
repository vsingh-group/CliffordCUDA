"""Gradcheck for the multi-k variant of geom_prod."""
import pytest

from cliffordcuda.extensions.ga.geom_prod import geom_prod_multik, load_geom_prod_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_geom_prod_multik(n, metric):
    _ = load_geom_prod_cuda()
    assert run_gradcheck(geom_prod_multik, n, metric)
