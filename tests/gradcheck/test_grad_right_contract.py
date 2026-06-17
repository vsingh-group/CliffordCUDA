"""Gradcheck for right_contract."""
import pytest

from cliffordcuda.extensions.ga.right_contract import right_contract, load_contract_cuda
from _gradcheck import all_cases, run_gradcheck, N_VALUES_FAST


@pytest.mark.parametrize("n,metric", all_cases(N_VALUES_FAST))
def test_grad_right_contract(n, metric):
    _ = load_contract_cuda()
    assert run_gradcheck(right_contract, n, metric)
