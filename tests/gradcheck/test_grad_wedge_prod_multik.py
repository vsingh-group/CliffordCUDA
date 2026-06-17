"""Gradcheck for wedge_prod_multik."""
import pytest

from cliffordcuda.extensions.ga.wedge_prod.chunk import wedge_prod_multik
from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_wedge_prod_multik(n, metric):
    _ = load_wedge_prod_cuda()
    assert run_gradcheck(wedge_prod_multik, n, metric)
