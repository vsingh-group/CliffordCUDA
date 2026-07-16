"""Matrix-representation inverse for non-degenerate Cl(p, q).

A non-degenerate Clifford algebra is a matrix algebra, Cl(p, q) ~= M_d(K) with
d = 2**floor(n/2) and K in {R, C, H}. So a multivector -- its 2**n coefficients
-- is really a d x d matrix, and its inverse is the (tiny) matrix inverse mapped
back. That replaces the O(8**n) dense solve on the 2**n x 2**n regular
representation with an O(4**n) map plus an O(d**3) inverse: ~100x faster at
n=13, and more accurate (a well-conditioned 64x64 solve instead of an 8192**2).

Construction: complex Dirac gammas gamma_i (d x d, Pauli-tensor products,
i-scaled for negative directions), all 2**n blade matrices Gamma_A built in n
batched matmuls (Gamma_{A|2^b} = Gamma_A @ gamma_b), then

    x  -> M(x) = sum_A x_A Gamma_A            (forward map)
    M(x)^-1                                    (small matrix inverse)
    x^-1_A = tr(Gamma_A^dagger M^-1) / d       (inverse map)

M is an algebra homomorphism, so M(x)^-1 = M(x^-1) and the round trip is exact.

Every non-degenerate signature is covered, in one of two modes (detected and
cached per (n, metric) by a round-trip probe):
  * "single": a single complex d x d block is faithful -- all even n, and odd n
    whose pseudoscalar squares to -1;
  * "two": the remaining odd signatures (e.g. Euclidean Cl(5,0), Cl(9,0),
    Cl(13,0)) are M_d(K) + M_d(K), split by the central pseudoscalar. A single
    block conflates each blade with its complement; running the second block
    (the grade involution) and averaging recovers everything. Two small inverses
    instead of one -- still ~40x faster than the dense LU at n=13.
Not applicable only when: the metric is degenerate (any 0 -> not a matrix
algebra), or the cached O(4**n) blade tensor exceeds the memory budget. In those
cases `inverse` falls back to the stable dense LU solve.
"""
import functools

import torch

from .geom_prod import geom_prod, _normalize_metric

_C = torch.complex64
_REP_MAX_BLADE_BYTES = 4e9          # cap on the cached (2**n, d, d) blade tensor


def _pauli(device):
    X = torch.tensor([[0, 1], [1, 0]], dtype=_C, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=_C, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=_C, device=device)
    I2 = torch.eye(2, dtype=_C, device=device)
    return X, Y, Z, I2


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def _build_gammas(n, metric, device):
    """n complex d x d gammas (d = 2**floor(n/2)) with gamma_i^2 = metric[i] I,
    pairwise anticommuting."""
    X, Y, Z, I2 = _pauli(device)
    m = n // 2
    d = 1 << m
    base = []
    for k in range(1, m + 1):
        left = [Z] * (k - 1)
        right = [I2] * (m - k)
        base.append(_kron_list(left + [X] + right))
        base.append(_kron_list(left + [Y] + right))
    if n % 2 == 1:
        base.append(_kron_list([Z] * m) if m > 0 else torch.eye(1, dtype=_C, device=device))
    gammas = []
    for i in range(n):
        g = base[i]
        if metric[i] == -1:
            g = 1j * g
        gammas.append(g.to(_C))
    return gammas, d


def _build_blade_mats(gammas, n, d, device):
    """All 2**n blade matrices, in n batched matmuls. Generator b is the new
    highest index, so Gamma_{A | 2^b} = Gamma_A @ gamma_b."""
    B = torch.eye(d, dtype=_C, device=device).view(1, d, d)
    for b in range(n):
        B = torch.cat([B, B @ gammas[b]], dim=0)
    return B


def _grade_sign(n, device):
    """(-1)^grade for every blade -- the grade involution's per-blade sign."""
    idx = torch.arange(1 << n, device=device)
    k = torch.zeros_like(idx)
    for b in range(n):
        k += (idx >> b) & 1
    return torch.where((k & 1) == 1, -1.0, 1.0).to(torch.float32)


def _map(x, B):
    """multivector -> matrix: M(x) = sum_A x_A Gamma_A."""
    return torch.einsum('bA,Aij->bij', x.to(_C), B)


def _unmap(M, Bh, d):
    """matrix -> multivector coefficients: x_A = tr(Gamma_A^dagger M) / d."""
    return torch.einsum('Aij,bji->bA', Bh, M).real / d


@functools.lru_cache(maxsize=None)
def _rep_tables(n, metric, device):
    """(B, Bh, d, mode, gsign) for a non-degenerate, in-budget signature, else
    None (caller uses LU).

    mode = "single": a single complex block M_d(K) is faithful (all even n, and
        odd n with omega^2 = -1).
    mode = "two": odd n with omega^2 = +1, where Cl(p,q) ~= A + A splits by the
        central pseudoscalar omega into two blocks. A single block conflates each
        blade with its complement; using the second block (the grade involution,
        gsign = (-1)^grade) and averaging recovers everything -- see rep_inverse.
    """
    if 0 in metric:                                   # degenerate: not a matrix algebra
        return None
    d = 1 << (n // 2)
    if (1 << n) * d * d * 8 > _REP_MAX_BLADE_BYTES:   # blade tensor too large
        return None
    gammas, d = _build_gammas(n, metric, device)
    # Clifford relations must hold (guards a bad construction).
    I = torch.eye(d, dtype=_C, device=device)
    for i in range(n):
        for j in range(n):
            anti = gammas[i] @ gammas[j] + gammas[j] @ gammas[i]
            want = (2.0 * metric[i]) * I if i == j else torch.zeros_like(I)
            if (anti - want).abs().max() > 1e-3:
                return None
    B = _build_blade_mats(gammas, n, d, device)
    Bh = B.conj().transpose(-1, -2).contiguous()
    probe = (torch.arange(1 << n, device=device, dtype=torch.float32) + 1.0).view(1, -1)
    tol = 1e-2 * probe.abs().max()
    # single-block round trip
    if (_unmap(_map(probe, B), Bh, d) - probe).abs().max() <= tol:
        return B, Bh, d, "single", None
    # two-block round trip: 0.5*(unmap(M+) + g * unmap(M-)), M- uses grade involution
    g = _grade_sign(n, device)
    back = 0.5 * (_unmap(_map(probe, B), Bh, d) + g * _unmap(_map(probe * g, B), Bh, d))
    if (back - probe).abs().max() <= tol:
        return B, Bh, d, "two", g
    return None                                        # neither -> LU


def rep_applicable(n, metric, device):
    return _rep_tables(n, _normalize_metric(n, metric), device) is not None


def _rep_inverse_core(x, metric):
    """map -> small inverse -> map (single or two block). No invertibility
    guard; raises torch.linalg.LinAlgError on an exactly singular block."""
    n = x.size(-1).bit_length() - 1
    B, Bh, d, mode, g = _rep_tables(n, _normalize_metric(n, metric), str(x.device))
    if mode == "single":
        xinv = _unmap(torch.linalg.inv(_map(x, B)), Bh, d)
    else:                                              # two-block: invert each block
        inv_p = _unmap(torch.linalg.inv(_map(x, B)), Bh, d)
        inv_m = _unmap(torch.linalg.inv(_map(x * g, B)), Bh, d)
        xinv = 0.5 * (inv_p + g * inv_m)
    # (.real is a strided view; make the result contiguous for downstream kernels)
    return xinv.contiguous()


def _guard_inverse(x, xinv, metric, atol):
    """Raise if x xinv is not (numerically) the scalar 1 -- catches singular /
    ill-conditioned inputs that produce finite garbage."""
    with torch.no_grad():
        prod = geom_prod(x, xinv, metric=metric)
        ref = torch.zeros_like(prod); ref[..., 0] = 1.0
        resid = (prod - ref).abs().amax(dim=-1)
    if not bool(torch.isfinite(resid).all()) or bool((resid > atol).any()):
        bad = int((~(resid <= atol)).sum())
        raise ValueError(
            f"{bad}/{x.size(0)} multivector(s) are not invertible or too "
            f"ill-conditioned (max residual {float(resid.max()):.2e}).")


def rep_inverse(x, metric, atol: float = 1e-2):
    """Inverse of x: (B, 2**n) via the matrix representation. Assumes
    rep_applicable(...) is True. Differentiable. Raises if x is not invertible."""
    try:
        xinv = _rep_inverse_core(x, metric)
    except torch.linalg.LinAlgError as e:
        raise ValueError(f"multivector is not invertible ({e}).")
    _guard_inverse(x, xinv, metric, atol)
    return xinv


@functools.lru_cache(maxsize=None)
def _degenerate_tables(n, metric, device):
    """(r, sub_metric, sub_idx, no_null) for a degenerate metric whose non-null
    part is representable, else None.

    A degenerate Cl(p,q,r) is a nilpotent extension of the non-degenerate
    Cl(p,q): split x = x0 (blades with no null generator) + eta (the rest); eta
    lives in the nilpotent ideal J with J^{r+1}=0. Then
        x^-1 = ( sum_{k=0}^{r} (-x0^-1 eta)^k ) x0^-1
    a terminating series -- x0 inverted by the rep on the smaller Cl(p,q), the
    corrections a handful of geometric products.
    """
    null_bits = [i for i in range(n) if metric[i] == 0]
    r = len(null_bits)
    if r == 0:
        return None
    nonnull = [i for i in range(n) if metric[i] != 0]
    n_sub = n - r
    sub_metric = tuple(metric[i] for i in nonnull)
    if not rep_applicable(n_sub, sub_metric, device):
        return None
    null_mask = 0
    for i in null_bits:
        null_mask |= (1 << i)
    # sub_idx[s] = the full-algebra blade index for sub-algebra blade s
    s = torch.arange(1 << n_sub, device=device)
    sub_idx = torch.zeros_like(s)
    for j, pos in enumerate(nonnull):
        sub_idx = sub_idx | (((s >> j) & 1) << pos)
    idx = torch.arange(1 << n, device=device)
    no_null = ((idx & null_mask) == 0).to(torch.float32)    # 1 for no-null blades
    return r, sub_metric, sub_idx, no_null


def degenerate_applicable(n, metric, device):
    return _degenerate_tables(n, _normalize_metric(n, metric), device) is not None


def degenerate_rep_inverse(x, metric, atol: float = 1e-2):
    """Inverse of x: (B, 2**n) in a degenerate Cl(p,q,r) via the nilpotent peel.
    Assumes degenerate_applicable(...) is True. Differentiable."""
    n = x.size(-1).bit_length() - 1
    metric = _normalize_metric(n, metric)
    r, sub_metric, sub_idx, no_null = _degenerate_tables(n, metric, str(x.device))

    x0 = x * no_null                                   # no-null part (in Cl(p,q))
    eta = x - x0                                        # null part (nilpotent ideal)
    x_sub = x0[:, sub_idx]                              # compress to Cl(p,q)
    try:
        inv_sub = _rep_inverse_core(x_sub, sub_metric)
    except torch.linalg.LinAlgError as e:
        raise ValueError(f"multivector is not invertible ({e}).")
    x0inv = torch.zeros_like(x).index_copy(1, sub_idx, inv_sub)   # embed back

    nu = geom_prod(x0inv, eta, metric=metric)          # nu = x0^-1 eta (nilpotent)
    one = torch.zeros_like(x); one[:, 0] = 1.0
    S = one
    for _ in range(r):                                 # Horner: sum_{k=0}^{r} (-nu)^k
        S = one - geom_prod(nu, S, metric=metric)
    xinv = geom_prod(S, x0inv, metric=metric).contiguous()
    _guard_inverse(x, xinv, metric, atol)
    return xinv
