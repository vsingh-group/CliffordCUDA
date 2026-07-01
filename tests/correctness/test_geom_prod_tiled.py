"""Correctness check for the tiled geometric product (`geom_prod_tiled`), which
handles n > 13 by splitting the multivector into shared-memory-sized tiles.

Two checks:
  (1) Forcing the tiling at n <= 13 (via `max_fit < n`) must reproduce the
      standard kernel exactly. This exercises the tiling, the sign
      factorization, the grade-involution cross term, and the accumulation
      against the trusted kernel, across Cl(p, q, r).
  (2) Past the standard kernel's n<=13 limit (n = 14), the result must not
      depend on the tile size: tiling with L=13 and L=12 must agree.
"""
import pytest
import torch

from cliffordcuda.extensions.ga.geom_prod import (
    geom_prod, build_packed_sign, _build_sign_value_table, _normalize_metric)
from cliffordcuda.extensions.ga.geom_prod_tiled import (
    geom_prod_tiled, geom_prod_tiled_fused, _sign_value_table_gpu, _pack_fwd_gpu)


# (n, metric, max_fit): max_fit < n forces the tiling path. Metrics span the
# Cl(p, q, r) space incl. degenerate and p,q,r all > 0.
FORCED = [
    (10, (1,) * 10, 5),                                  # Cl(10,0,0)
    (10, (1, 1, 1, 1, 1, -1, -1, -1, -1, -1), 7),        # Cl(5,5,0)
    (10, (1, 1, 1, -1, -1, 0, 0, 1, -1, 0), 9),          # Cl(4,3,3) -- p,q,r all > 0
    (11, (1,) * 11, 6),                                  # Cl(11,0,0)
    (11, (1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1), 9),     # Cl(6,5,0)
    (12, (1, 1, -1, -1, 1, -1, 0, 0, 1, 1, -1, 0), 7),   # Cl(5,4,3) -- p,q,r all > 0
    (13, (1,) * 13, 10),                                 # Cl(13,0,0)
]


@pytest.mark.parametrize("n,metric,max_fit", FORCED, ids=lambda v: str(v))
def test_tiled_matches_standard(n, metric, max_fit):
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    ref   = geom_prod(a, b, metric=metric)
    tiled = geom_prod_tiled(a, b, metric=metric, max_fit=max_fit)
    diff = float((tiled - ref).abs().max().item())
    assert diff < 1e-3, f"|tiled - standard| = {diff} (n={n}, max_fit={max_fit})"


# Past the standard kernel's n<=13 limit: result must be independent of tile size.
PAST_LIMIT = [
    (14, (1,) * 14),                       # Cl(14,0,0)
    (14, (1,) * 12 + (-1, -1)),            # Cl(12,2,0)
    (14, (1,) * 11 + (-1, -1, 0)),         # Cl(11,2,1) -- p,q,r all > 0
]


@pytest.mark.parametrize("n,metric", PAST_LIMIT, ids=lambda v: str(v))
def test_tiled_past_limit_consistent(n, metric):
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    t13 = geom_prod_tiled(a, b, metric=metric, max_fit=13)
    t12 = geom_prod_tiled(a, b, metric=metric, max_fit=12)
    diff = float((t13 - t12).abs().max().item())
    assert diff < 1e-3, f"L=13 vs L=12 tilings disagree: {diff} (n={n})"


# ── Fused single-kernel variant ──────────────────────────────────────────────

@pytest.mark.parametrize("n,metric,max_fit", FORCED, ids=lambda v: str(v))
def test_fused_matches_standard(n, metric, max_fit):
    """Fused kernel, forced to tile (max_fit < n) at n <= 13, must reproduce the
    standard kernel across Cl(p, q, r)."""
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    ref   = geom_prod(a, b, metric=metric)
    fused = geom_prod_tiled_fused(a, b, metric=metric, max_fit=max_fit)
    diff = float((fused - ref).abs().max().item())
    assert diff < 1e-3, f"|fused - standard| = {diff} (n={n}, max_fit={max_fit})"


@pytest.mark.parametrize("n,metric", PAST_LIMIT, ids=lambda v: str(v))
def test_fused_matches_orchestrated_past_limit(n, metric):
    """Past the standard kernel's limit, the auto-L fused kernel must match the
    independently-verified orchestrated tiling."""
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    fused = geom_prod_tiled_fused(a, b, metric=metric)
    orch  = geom_prod_tiled(a, b, metric=metric)
    diff = float((fused - orch).abs().max().item())
    assert diff < 1e-3, f"|fused - orchestrated| = {diff} (n={n})"


# GPU-built sign LUT must be bit-identical to the CPU builder, across Cl(p,q,r).
LUT_CASES = [
    (10, (1,) * 10),                                     # Cl(10,0,0)
    (11, (-1,) * 11),                                    # Cl(0,11,0)
    (12, (1, 1, 1, -1, -1, 0, 0, 1, -1, 0, 1, -1)),      # Cl(4,4,4) -- p,q,r all > 0
    (13, (1,) * 11 + (-1, 0)),                           # Cl(11,1,1)
]


@pytest.mark.parametrize("L,metric", LUT_CASES, ids=lambda v: str(v))
def test_gpu_lut_bit_exact(L, metric):
    metric = _normalize_metric(L, metric)
    table_gpu = _sign_value_table_gpu(L, metric, 'cuda')
    assert torch.equal(table_gpu.cpu(), _build_sign_value_table(L, metric)), "sigma table differs"
    ps_c, pv_c = build_packed_sign(L, 'cuda', metric)
    ps_g, pv_g = _pack_fwd_gpu(table_gpu, 'cuda')
    assert torch.equal(ps_g.cpu(), ps_c.cpu()), "packed_sign differs"
    if pv_c is None:
        assert pv_g is None, "packed_valid should be None for non-degenerate metric"
    else:
        assert torch.equal(pv_g.cpu(), pv_c.cpu()), "packed_valid differs"


# Public geom_prod must transparently work past the standard kernel's limit.
@pytest.mark.parametrize("n,metric", PAST_LIMIT, ids=lambda v: str(v))
def test_public_geom_prod_past_limit(n, metric):
    dim = 1 << n
    torch.manual_seed(0)
    a = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    b = torch.randn(2, dim, device='cuda', dtype=torch.float32)
    diff = float((geom_prod(a, b, metric=metric) - geom_prod_tiled(a, b, metric=metric)).abs().max())
    assert diff < 1e-3, f"public geom_prod vs orchestrated tiled: {diff} (n={n})"


# The fused kernel's autograd backward must match the (differentiable) orchestrated
# path, which is correct by construction -- across batch sizes and Cl(p, q, r).
GRAD_CASES = [
    (14, (1,) * 14, 1),                        # Cl(14,0,0)
    (14, (1,) * 11 + (-1, -1, 0), 4),          # Cl(11,2,1) degenerate, batched
    (15, (1,) * 9 + (-1,) * 6, 2),             # Cl(9,6,0) batched
    (16, (1,) * 13 + (-1, -1, 0), 1),          # Cl(13,2,1) degenerate
]


@pytest.mark.parametrize("n,metric,B", GRAD_CASES, ids=lambda v: str(v))
def test_fused_autograd_matches_orchestrated(n, metric, B):
    dim = 1 << n
    torch.manual_seed(0)
    a0 = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    b0 = torch.randn(B, dim, device='cuda', dtype=torch.float32)
    g = torch.randn(B, dim, device='cuda', dtype=torch.float32)

    def grads(fn):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        (fn(a, b) * g).sum().backward()
        return a.grad, b.grad

    fa, fb = grads(lambda a, b: geom_prod_tiled_fused(a, b, metric=metric))
    oa, ob = grads(lambda a, b: geom_prod_tiled(a, b, metric=metric))
    assert float((fa - oa).abs().max()) < 1e-3, f"grad_a mismatch (n={n}, B={B})"
    assert float((fb - ob).abs().max()) < 1e-3, f"grad_b mismatch (n={n}, B={B})"
