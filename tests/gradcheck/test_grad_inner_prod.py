"""Gradcheck for inner_prod (chunk variant)."""
import pytest

from cliffordcuda.extensions.ga.inner_prod import inner_prod, load_inner_prod_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_inner_prod(n, metric):
    _ = load_inner_prod_cuda()
    assert run_gradcheck(inner_prod, n, metric)
