"""Correctness for the multivector inverse (`inverse`, `CliffordAlgebra.inverse`).

Two paths are exercised:
  * versor/blade fast path `x~/(x x~)` -- exact at every n (incl. n where the
    dense matrix would not fit), checked both sides `x x^-1` and `x^-1 x`;
  * general dense solve `L y = e_0` -- checked against an independent matrix
    inverse across Cl(p, q, r) (incl. degenerate and p,q,r all > 0) and batch.

Autograd is smoke-checked here (grad finite, both paths); the differentiable
structure is verified rigorously by an fp64 gradcheck in development.
"""
import pytest
import torch

from cliffordcuda.algebra import CliffordAlgebra
from cliffordcuda.extensions.ga.geom_prod import geom_prod, _build_sign_value_table
from cliffordcuda.extensions.ga.inverse import inverse
from cliffordcuda.extensions.ga.inverse_rep import rep_applicable, degenerate_applicable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _scalar_res(prod):
    ref = torch.zeros_like(prod)
    ref[..., 0] = 1.0
    return float((prod - ref).abs().max().item())


def _rand_vector(B, n):
    """A grade-1 multivector (its geometric square is a scalar -> a versor)."""
    x = torch.zeros(B, 1 << n, device=DEV)
    for i in range(n):
        x[:, 1 << i] = torch.randn(B, device=DEV)
    return x


def _matrix_inverse(x, n, metric):
    """Independent ground truth: L[k,j] = sigma(j^k, j) x[j^k], solve L y = e_0."""
    dim = 1 << n
    sv = _build_sign_value_table(n, metric).to(DEV).float()
    kk = torch.arange(dim, device=DEV).view(dim, 1)
    jj = torch.arange(dim, device=DEV).view(1, dim)
    a = jj ^ kk
    S = sv[a, jj]
    outs = []
    for b in range(x.size(0)):
        L = S * x[b][a]
        e0 = torch.zeros(dim, device=DEV); e0[0] = 1.0
        outs.append(torch.linalg.solve(L, e0))
    return torch.stack(outs, 0)


# Metrics spanning Cl(p, q, r): Euclidean, indefinite, and degenerate with
# p, q, r all > 0.
def _metrics(n):
    out = [((1,) * n, "eucl"),
           (tuple(-1 if i % 3 == 0 else 1 for i in range(n)), "mixed")]
    if n >= 3:
        out.append(((0,) + (1,) * (n - 2) + (-1,), "deg"))
    return out


@pytest.mark.parametrize("n", [5, 8, 11, 13, 15, 17])
def test_versor_exact_all_n(n):
    """Versor inverse is exact at every n, both sides."""
    torch.manual_seed(0)
    for metric, _ in _metrics(n):
        R = geom_prod(_rand_vector(2, n), _rand_vector(2, n), metric=metric)
        inv = inverse(R, metric=metric)
        assert _scalar_res(geom_prod(R, inv, metric=metric)) < 1e-4
        assert _scalar_res(geom_prod(inv, R, metric=metric)) < 1e-4


@pytest.mark.parametrize("n", [5, 7, 9, 10])
@pytest.mark.parametrize("B", [1, 3])
def test_general_matches_matrix_solve(n, B):
    """General (non-versor) inverse matches an independent matrix inverse."""
    for metric, _ in _metrics(n):
        torch.manual_seed(100 + n)
        x = torch.randn(B, 1 << n, device=DEV)
        x[..., 0] += 3.0                              # scalar-dominant -> invertible
        inv = inverse(x, metric=metric, method="lu")
        gt = _matrix_inverse(x, n, metric)
        assert (inv - gt).abs().max().item() < 1e-3
        # and it really is a right inverse
        assert _scalar_res(geom_prod(x, inv, metric=metric)) < 1e-2


def test_auto_dispatch_picks_versor_then_lu():
    n = 8
    metric = (1,) * n
    torch.manual_seed(0)
    R = geom_prod(_rand_vector(1, n), _rand_vector(1, n), metric=metric)
    g = torch.randn(1, 1 << n, device=DEV); g[..., 0] += 3.0
    # versor -> exact fast path (much tighter than any LU residual)
    assert _scalar_res(geom_prod(R, inverse(R, metric=metric), metric=metric)) < 1e-6
    # general -> LU still a valid inverse
    assert _scalar_res(geom_prod(g, inverse(g, metric=metric), metric=metric)) < 1e-2


def test_method_versor_rejects_non_versor():
    n = 6
    torch.manual_seed(0)
    g = torch.randn(1, 1 << n, device=DEV); g[..., 0] += 3.0
    with pytest.raises(ValueError, match="not a versor|not scalar"):
        inverse(g, metric=(1,) * n, method="versor")


def test_non_invertible_raises():
    n = 6
    # zero multivector (versor path: x x~ = 0)
    with pytest.raises(ValueError, match="not invertible"):
        inverse(torch.zeros(1, 1 << n, device=DEV), metric=(1,) * n)
    # a null blade in a degenerate metric
    x = torch.zeros(1, 1 << n, device=DEV); x[0, 1] = 1.0
    with pytest.raises(ValueError, match="not invertible"):
        inverse(x, metric=(0,) + (1,) * (n - 1), method="versor")


def test_leading_dims_preserved():
    n = 6
    metric = (1,) * n
    torch.manual_seed(0)
    x = torch.randn(3, 4, 1 << n, device=DEV); x[..., 0] += 3.0
    inv = inverse(x, metric=metric)
    assert inv.shape == x.shape
    flat_r = _scalar_res(geom_prod(x.reshape(-1, 1 << n),
                                   inv.reshape(-1, 1 << n), metric=metric))
    assert flat_r < 1e-2


@pytest.mark.parametrize("method", ["versor", "lu", "rep"])
def test_autograd_grad_finite(method):
    n = 8                                              # even n -> rep applicable
    metric = (1,) * n
    torch.manual_seed(0)
    if method == "versor":
        x = geom_prod(_rand_vector(1, n), _rand_vector(1, n), metric=metric)
    else:
        x = torch.randn(1, 1 << n, device=DEV); x[..., 0] += 3.0
    x = x.clone().requires_grad_(True)
    inverse(x, metric=metric, method=method).pow(2).sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert x.grad.abs().max().item() > 0


# (n, metric, expected rep_applicable): every non-degenerate signature is
# representable (single or two block); only degenerate metrics are not.
REP_DISPATCH = [
    (7, (1,) * 7, True),                  # Cl(7,0): single block
    (8, (1,) * 8, True),                  # even n: single block
    (5, (1, 1, 1, 1, -1), True),          # Cl(4,1): single block
    (5, (1,) * 5, True),                  # Cl(5,0): two block
    (9, (1,) * 9, True),                  # Cl(9,0): two block
    (13, (1,) * 13, True),                # Cl(13,0): two block
    (9, (-1,) + (1,) * 8, True),          # Cl(8,1): single block
    (6, (0,) + (1,) * 5, False),          # degenerate -> LU
]


@pytest.mark.parametrize("n,metric,expected", REP_DISPATCH,
                         ids=lambda v: str(v))
def test_rep_applicable_matches_signature(n, metric, expected):
    assert rep_applicable(n, metric, "cuda") is expected


# single-block and two-block signatures both agree with LU.
@pytest.mark.parametrize("n,metric", [(7, (1,) * 7), (8, (1,) * 8),
                                      (5, (1, 1, 1, 1, -1)), (9, (-1,) + (1,) * 8),
                                      (10, (1,) * 10),
                                      (5, (1,) * 5), (9, (1,) * 9)])  # two-block
def test_rep_matches_lu(n, metric):
    """Where the rep applies, it agrees with the dense LU and is a valid inverse."""
    assert rep_applicable(n, metric, "cuda")
    torch.manual_seed(n)
    x = torch.randn(3, 1 << n, device=DEV); x[..., 0] += 3.0
    inv_rep = inverse(x, metric=metric, method="rep")
    inv_lu = inverse(x, metric=metric, method="lu")
    assert (inv_rep - inv_lu).abs().max().item() < 1e-2
    assert _scalar_res(geom_prod(x, inv_rep, metric=metric)) < 1e-2
    assert inv_rep.is_contiguous()


def test_two_block_euclidean_uses_rep_not_lu():
    """Odd Euclidean signatures (Cl(9,0)) are covered by the two-block rep."""
    n, metric = 9, (1,) * 9
    assert rep_applicable(n, metric, "cuda")
    torch.manual_seed(0)
    x = torch.randn(2, 1 << n, device=DEV); x[..., 0] += 3.0
    inv = inverse(x, metric=metric, method="rep")     # must not raise
    assert _scalar_res(geom_prod(x, inv, metric=metric)) < 1e-2


# degenerate metrics whose non-null part is representable: nilpotent-peel path.
DEGEN = [
    (6, (0,) + (1,) * 5),                 # Cl(5,0,1): r=1, sub Cl(5,0) two-block
    (7, (0,) + (1,) * 6),                 # Cl(6,0,1): r=1, sub Cl(6,0) single
    (6, (1, 1, 1, 1, -1, 0)),             # Cl(3,1,1): r=1, mixed
    (7, (0, 0) + (1,) * 5),               # Cl(5,0,2): r=2
]


@pytest.mark.parametrize("n,metric", DEGEN, ids=lambda v: str(v))
def test_degenerate_matches_lu(n, metric):
    """Degenerate metrics invert via the nilpotent peel and agree with LU."""
    assert degenerate_applicable(n, metric, "cuda")
    assert not rep_applicable(n, metric, "cuda")
    torch.manual_seed(n)
    x = torch.randn(3, 1 << n, device=DEV); x[..., 0] += 3.0
    inv = inverse(x, metric=metric)                   # auto -> degenerate peel
    inv_lu = inverse(x, metric=metric, method="lu")
    assert (inv - inv_lu).abs().max().item() < 1e-2
    assert _scalar_res(geom_prod(x, inv, metric=metric)) < 1e-2


def test_degenerate_autograd_grad_finite():
    n, metric = 6, (0,) + (1,) * 5
    torch.manual_seed(0)
    x = torch.randn(1, 1 << n, device=DEV); x[..., 0] += 3.0
    x = x.clone().requires_grad_(True)
    inverse(x, metric=metric).pow(2).sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert x.grad.abs().max().item() > 0


def test_clifford_algebra_inverse_method():
    ca = CliffordAlgebra((1, 1, 1, -1, 1))          # Cl(4, 1), n = 5
    torch.manual_seed(0)
    R = geom_prod(_rand_vector(4, 5), _rand_vector(4, 5), metric=ca.metric)
    inv = ca.inverse(R)
    assert _scalar_res(ca.geom_prod(R, inv)) < 1e-5
    # general path through the class method
    g = torch.randn(2, ca.dim, device=DEV); g[..., 0] += 3.0
    inv_g = ca.inverse(g)
    assert _scalar_res(ca.geom_prod(g, inv_g)) < 1e-2
