"""Inverse of a general multivector in Cl(p, q, r).

Three paths, selected automatically by `inverse(..., method="auto")`:

  1. Versor / blade fast path.  When ``x x~`` is a scalar (true for every
     versor and blade -- rotors, motors, reflections, single blades), the
     inverse is exactly

         x^-1 = x~ / (x x~)

     one reversion, one geometric product, one scalar divide.  Exact in fp32
     at *every* n (it never builds a matrix), and it is the common case.  This
     is the primary path.

  2. General multivector -- matrix representation (`inverse_rep`).  A
     non-degenerate Cl(p, q) is a matrix algebra, so a multivector is really a
     small matrix (side d = 2**floor(n/2)); map to it, invert the small matrix,
     map back.  ~40-100x faster than the dense solve at n=13 and more accurate
     (a 64x64 inverse instead of an 8192**2). Covers every non-degenerate
     signature (one or two blocks). A *degenerate* metric is a nilpotent
     extension of the non-degenerate Cl(p, q): split off the null part and use a
     terminating series over the rep inverse of the smaller Cl(p, q) -- also
     ~40-50x faster than LU. Falls through to (3) only if the relevant blade
     tensor does not fit.

  3. General multivector, fallback -- dense linear solve.  Left-multiplication
     by ``x`` is the matrix ``L[k, j] = sigma(j^k, j) x[j^k]``, and ``x^-1``
     solves ``L y = e_0``.  Pivoted LU (``torch.linalg.solve``): stable and
     accurate in fp32 (~1e-5 residual), O(4**n) memory (feasible to ~n=14).
     Used only when the representation does not fit (n too large, or a
     degenerate metric whose non-null part is too large).

The Faddeev-LeVerrier / Shirokov recursion is deliberately not used: building a
characteristic polynomial from traces of powers is ill-conditioned and unusable
past small n in any precision, whereas paths (2)/(3) are stable in fp32.

All paths are differentiable w.r.t. x.
"""
import functools

import torch

from .geom_prod import geom_prod, _normalize_metric
from .geom_prod_tiled import _sign_value_table_gpu
from .inverse_rep import (
    rep_inverse, rep_applicable, degenerate_rep_inverse, degenerate_applicable)


def _popcount(idx: torch.Tensor, n: int) -> torch.Tensor:
    k = torch.zeros_like(idx)
    for b in range(n):
        k += (idx >> b) & 1
    return k


@functools.lru_cache(maxsize=None)
def _reverse_signs(n: int, device: str) -> torch.Tensor:
    """Per-blade +/-1 for reversion x~: grade k -> (-1)^(k(k-1)/2)."""
    idx = torch.arange(1 << n, device=device)
    k = _popcount(idx, n)
    return torch.where(((k * (k - 1) // 2) % 2) == 1, -1.0, 1.0).to(torch.float32)


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Reversion x~ (reverse the order of basis vectors in every blade)."""
    n = x.size(-1).bit_length() - 1
    return x * _reverse_signs(n, str(x.device))


def _versor_inverse(x: torch.Tensor, metric, rtol: float):
    """Try the exact versor/blade inverse x~/(x x~). Returns (inv, scalar, ok);
    ok is False if x x~ is not (numerically) a scalar, i.e. x is not a versor."""
    r = reverse(x)
    xr = geom_prod(x, r, metric=metric)                       # x x~
    s = xr[..., :1]                                           # scalar part
    nonscalar = xr[..., 1:].abs().amax(dim=-1, keepdim=True)  # largest off-scalar
    scale = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    ok = bool((nonscalar <= rtol * scale).all())
    return r / s, s, ok


@functools.lru_cache(maxsize=None)
def _lu_tables(n: int, metric: tuple, device: str):
    """Cached (S, gather_idx) for the dense solve.
      gather_idx[k, j] = j ^ k          (input coefficient feeding L[k, j])
      S[k, j]          = sigma(j^k, j)  (the fixed sign matrix)
    so that L[.., k, j] = S[k, j] * x[.., j^k]."""
    dim = 1 << n
    sv = _sign_value_table_gpu(n, metric, device)             # (dim, dim) sigma(a, b)
    kk = torch.arange(dim, device=device).view(dim, 1)
    jj = torch.arange(dim, device=device).view(1, dim)
    a = jj ^ kk                                               # (dim, dim) = j^k
    S = sv[a, jj].to(torch.float32)                           # sigma(j^k, j)
    return S, a


def _lu_inverse(xf: torch.Tensor, metric, atol: float = 1e-2) -> torch.Tensor:
    """General inverse of xf: (B, dim) via the stable dense solve L y = e_0.
    Differentiable. Rejects non-invertible / hopelessly ill-conditioned inputs
    by verifying the (right) residual ||x x^-1 - 1||."""
    B, dim = xf.shape
    n = dim.bit_length() - 1

    need = 2 * B * dim * dim * xf.element_size()              # matrix + LU workspace
    if xf.is_cuda:
        free, _ = torch.cuda.mem_get_info(xf.device)
        if need > 0.8 * free:
            raise ValueError(
                f"general (non-versor) inverse needs a dense ({B}, 2**{n}, 2**{n}) "
                f"system (~{need/1e9:.1f} GB) but only {free/1e9:.1f} GB is free. "
                f"If x is a versor/blade use method='versor' (no matrix); otherwise "
                f"reduce batch or n.")

    S, a = _lu_tables(n, tuple(metric), str(xf.device))
    L = S * xf[:, a]                                          # (B, dim, dim)
    rhs = torch.zeros(B, dim, 1, device=xf.device, dtype=xf.dtype)
    rhs[:, 0, 0] = 1.0
    try:
        y = torch.linalg.solve(L, rhs).squeeze(-1)            # (B, dim)
    except torch.linalg.LinAlgError as e:
        raise ValueError(f"multivector is not invertible ({e}).")

    # Verify it is actually an inverse (catches near-singular finite garbage,
    # which a plain isfinite check misses). Residual grows for singular / very
    # ill-conditioned x; good fp32 inverses are ~1e-4 to n=14. This check is a
    # guard only -- it must not contribute to y's gradient.
    with torch.no_grad():
        prod = geom_prod(xf, y, metric=metric)
        resid = (prod - rhs.squeeze(-1)).abs().amax(dim=-1)   # per multivector
    if not bool(torch.isfinite(resid).all()) or bool((resid > atol).any()):
        bad = int((~(resid <= atol)).sum())
        raise ValueError(
            f"{bad}/{B} multivector(s) are not invertible or are too "
            f"ill-conditioned for a stable fp32 inverse (max residual "
            f"{float(resid.max()):.2e}).")
    return y


def inverse(x: torch.Tensor, metric=None, method: str = "auto",
            rtol: float = 1e-4) -> torch.Tensor:
    """Inverse ``x^-1`` of a multivector in Cl(p, q, r).

    x: (..., 2**n) real multivector(s), bit-pattern component order.
    metric: length-n tuple in {-1, 0, 1}; None -> Cl(n, 0).
    method:
        "auto"   - versor fast path when ``x x~`` is scalar, else the matrix
                   representation when the signature supports it, else the dense
                   LU solve.
        "versor" - force ``x~/(x x~)``; errors if ``x x~`` is not scalar.
        "rep"    - force the matrix-representation inverse; errors if the
                   signature is degenerate or not representable.
        "lu"     - force the dense solve ``L y = e_0``.
    rtol: relative tolerance for the "is ``x x~`` scalar?" versor test.

    Raises ValueError if ``x`` is not invertible, or if the dense solve is
    needed but the ``(batch, 2**n, 2**n)`` system does not fit in memory.
    Differentiable w.r.t. x.
    """
    dim = x.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric = _normalize_metric(n, metric)
    lead = x.shape[:-1]
    xf = x.reshape(-1, dim)                       # (B, dim): the kernels are 2D-only

    if method in ("auto", "versor"):
        inv_v, s, ok = _versor_inverse(xf, metric, rtol)
        if method == "versor" and not ok:
            raise ValueError(
                "method='versor' but x x~ is not scalar (x is not a versor/"
                "blade); use method='auto', 'rep', or 'lu' for a general inverse.")
        if ok:
            if bool((s.abs() < 1e-20).any()):
                raise ValueError("multivector is not invertible (x x~ = 0).")
            return inv_v.reshape(*lead, dim)
    elif method not in ("lu", "rep"):
        raise ValueError(
            f"method must be 'auto', 'versor', 'rep', or 'lu'; got {method!r}")

    # General multivector: matrix representation when the signature supports it
    # (any non-degenerate metric, or a degenerate one whose non-null part is
    # representable), else the stable dense LU solve.
    dev = str(xf.device)
    if method in ("auto", "rep"):
        if rep_applicable(n, metric, dev):
            return rep_inverse(xf, metric).reshape(*lead, dim)
        if degenerate_applicable(n, metric, dev):
            return degenerate_rep_inverse(xf, metric).reshape(*lead, dim)
        if method == "rep":
            raise ValueError(
                "method='rep' but this signature is not representable "
                "(non-null part too large, or n too large); use 'auto' or 'lu'.")
    return _lu_inverse(xf, metric).reshape(*lead, dim)
