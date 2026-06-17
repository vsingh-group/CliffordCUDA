"""Gradcheck for left_contract."""
import pytest

from cliffordcuda.extensions.ga.left_contract import left_contract, load_contract_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_left_contract(n, metric):
    _ = load_contract_cuda()
    assert run_gradcheck(left_contract, n, metric)
