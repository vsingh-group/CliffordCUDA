"""Gradcheck for inner_prod_multik."""
import pytest

from cliffordcuda.extensions.ga.inner_prod.chunk import inner_prod_multik
from cliffordcuda.extensions.ga.inner_prod import load_inner_prod_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_inner_prod_multik(n, metric):
    _ = load_inner_prod_cuda()
    assert run_gradcheck(inner_prod_multik, n, metric)
