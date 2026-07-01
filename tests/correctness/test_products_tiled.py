"""Tiled fused forward+backward for the non-geometric products (wedge, inner,
left/right contraction, regressive), which handle n beyond the standard kernel's
device-dependent shared-memory limit.

Each tiled fused product reuses the geometric-product tiled kernel with the
product's masked sign tables (wedge/left/right are direct masks; inner =
left + right - diag; regressive = dual(dual(a) wedge dual(b))).

Checks, across Cl(p, q, r) and batch sizes:
  (1) Forcing the tiling at n <= 13 (max_fit < n) reproduces the standard
      differentiable kernel -- forward AND both gradients.
  (2) The public API transparently tiles past n=13 (forward + grad).
"""
import pytest
import torch

from cliffordcuda.extensions.ga.wedge_prod import wedge_prod
from cliffordcuda.extensions.ga.inner_prod import inner_prod
from cliffordcuda.extensions.ga.left_contract import left_contract
from cliffordcuda.extensions.ga.right_contract import right_contract
from cliffordcuda.extensions.ga.regressive_prod import regressive_prod
from cliffordcuda.extensions.ga.geom_prod_tiled import (
    wedge_prod_tiled_fused, inner_prod_tiled_fused, left_contract_tiled_fused,
    right_contract_tiled_fused, regressive_prod_tiled_fused,
)

# (id, standard_fn, tiled_fused_fn, allow_degenerate)
PRODUCTS = [
    ("wedge", wedge_prod,      wedge_prod_tiled_fused,      True),
    ("left",  left_contract,   left_contract_tiled_fused,   True),
    ("right", right_contract,  right_contract_tiled_fused,  True),
    ("inner", inner_prod,      inner_prod_tiled_fused,      True),
    ("regr",  regressive_prod, regressive_prod_tiled_fused, False),  # non-degenerate only
]
PROD_IDS = [p[0] for p in PRODUCTS]


def _metric(name, n):
    if name == "eucl":
        return (1,) * n
    if name == "neg":
        return (-1,) * n
    if name == "mix":                                  # Cl(p, q, 0), p,q > 0
        return tuple((1, -1, 1, -1)[i % 4] for i in range(n))
    if name == "deg":                                  # Cl(p, q, r), r > 0
        return tuple((1, 1, 1, -1, -1, 0)[i % 6] for i in range(n))
    raise ValueError(name)


CASES = [(m, B) for m in ("eucl", "neg", "mix", "deg") for B in (1, 3)]


def _grads(fn, a0, b0, g):
    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    (fn(a, b) * g).sum().backward()
    return a.grad, b.grad


@pytest.mark.parametrize("pid,std,tiled,allow_deg", PRODUCTS, ids=PROD_IDS)
@pytest.mark.parametrize("metric_name,B", CASES, ids=lambda v: str(v))
def test_product_tiled_matches_standard(pid, std, tiled, allow_deg, metric_name, B):
    """Forced tiling (n=11, L=7) reproduces the standard kernel, fwd + grads."""
    if metric_name == "deg" and not allow_deg:
        pytest.skip("regressive requires a non-degenerate metric")
    n, max_fit = 11, 7
    metric = _metric(metric_name, n)
    dim = 1 << n
    torch.manual_seed(0)
    a0 = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    b0 = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    g = torch.randn(B, dim, device='cuda', dtype=torch.float32)

    fwd = float((tiled(a0, b0, metric=metric, max_fit=max_fit)
                 - std(a0, b0, metric=metric)).abs().max())
    assert fwd < 1e-3, f"{pid} fwd {metric_name} B={B}: {fwd}"

    ta, tb = _grads(lambda a, b: tiled(a, b, metric=metric, max_fit=max_fit), a0, b0, g)
    ra, rb = _grads(lambda a, b: std(a, b, metric=metric), a0, b0, g)
    assert float((ta - ra).abs().max()) < 1e-3, f"{pid} grad_a {metric_name} B={B}"
    assert float((tb - rb).abs().max()) < 1e-3, f"{pid} grad_b {metric_name} B={B}"


@pytest.mark.parametrize("pid,std,tiled,allow_deg", PRODUCTS, ids=PROD_IDS)
def test_product_public_dispatch_past_limit(pid, std, tiled, allow_deg):
    """The public product transparently tiles past n=13 (forward + differentiable)."""
    n = 14
    metric = _metric("eucl", n)
    dim = 1 << n
    torch.manual_seed(0)
    a0 = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    b0 = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    diff = float((std(a0, b0, metric=metric) - tiled(a0, b0, metric=metric)).abs().max())
    assert diff < 1e-3, f"{pid} public-dispatch vs tiled: {diff}"

    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    std(a, b, metric=metric).square().sum().backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad).all())
    assert b.grad is not None and bool(torch.isfinite(b.grad).all())
