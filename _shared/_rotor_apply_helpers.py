"""Wrapper classes for the rotor-application benchmark.

Implementations of "apply the sandwich rotor parameterised by a bivector
to a multivector x" — used by `benchmark/ga/{speed,memory}/bench_rotor_apply*.py`
and `tests/correctness/test_rotor_apply.py`.

Each module exposes the same interface:
    model.bivectors_left   — nn.Parameter, shape (1, num_basis_biv), lex-pair order
    model._update_rotors() — rebuild the internal "rep" (matrix / rotor / cs)
    model(x)               — forward (FWD = inference, uses prebuilt rep)
                             when training (.train() + grad enabled), rep is
                             rebuilt inside forward where the impl is set up
                             that way (chunk does this; TorchGA / Versor
                             require an explicit _update_rotors() between
                             gradient updates).

Bivector convention: lex-pair order (e_01, e_02, ..., e_(n-2)(n-1)) over n
basis vectors. This matches torch_ga's ShortLex grade-2 region and the order
torch.triu_indices(n, n, offset=1) returns.

The chunk impl (CliffordAlgebra.compile_bivector + apply_rotor) lives in
the cliffordcuda package itself; it isn't redefined here — the bench just
imports CliffordAlgebra directly.
"""
import math

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra as VersorAlgebra
from torch_ga.mv_ops import mv_multiply, mv_reversion


# ---------------------------------------------------------------------------
# Convention helpers
# ---------------------------------------------------------------------------

def _lex_pair_to_bp_indices(n: int) -> torch.Tensor:
    """Returns a 1-D tensor of bit-pattern blade indices, one per lex-pair
    bivector slot, in the same order torch.triu_indices(n, n, offset=1)
    produces (i.e. (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2, n-1))."""
    skew_i, skew_j = torch.triu_indices(n, n, offset=1)
    return (1 << skew_i) | (1 << skew_j)


# ---------------------------------------------------------------------------
# Impl 1: torch_ga
# ---------------------------------------------------------------------------

class TorchGARotor(nn.Module):
    """Rotor application using only upstream torch_ga primitives.

    For the bench we don't construct R from a bivector via `exp` — upstream
    `approx_exp` overflows fp32 on a randn bivector at n>=9, and running it
    in fp64 on CPU (the only verbatim path that bypasses upstream's
    CPU-side `i_factorial` bug) is impractically slow. Instead we draw R
    directly as a random element of the subeven algebra (even-grade
    blades). It isn't a unit rotor but it has the right blade-support
    structure for the timed sandwich, which is what this bench is for.

    Why we don't instantiate `GeometricAlgebra`: upstream's `__init__`
    calls `cayley.py:get_cayley_tensor`, a pure-Python triple loop with
    O(D^3) Python work (uses `blades.index()` per pair). At n=11 this
    takes minutes. The timed path itself only needs `mv_multiply` and
    `mv_reversion` (the same functions `algebra.geom_prod` /
    `algebra.reversion` wrap internally), so we call those directly with
    a vectorized-built Cayley + blade-degree tensor.

    Setup (cached per `(n, device, dtype)`, outside the timed path):
      _cayley         : (D, D, D) dense geom Cayley (our vectorized builder)
      _blade_degrees  : (D,) grade of each ShortLex blade
      _even_mask      : (D,) bool, where blade_degrees % 2 == 0
    _update_rotors:
      R     = randn(D) masked to _even_mask
      R_rev = mv_reversion(R, blade_degrees)             # upstream function
    forward:
      y = mv_multiply(R_rev, x, cayley)                   # upstream function
      y = mv_multiply(y, R, cayley)                       # upstream function
    """
    _state = {}     # (n, device, dtype) -> dict(cayley, blade_degrees)

    def __init__(self, dim, device='cuda', dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.n = int(math.log2(dim))
        self.num_basis_biv = math.comb(self.n, 2)

        key = (self.n, str(torch.device(device)), dtype)
        if TorchGARotor._state.get(key) is None:
            TorchGARotor._state[key] = self._build_state(self.n, device, dtype)
        state = TorchGARotor._state[key]
        self._cayley        = state["cayley"]
        self._blade_degrees = state["blade_degrees"]

        # Kept for API symmetry with the other rotor wrappers (the rotor
        # benches don't overwrite this — R is drawn from a random even
        # multivector instead).
        self.bivectors_left = nn.Parameter(
            torch.randn(1, self.num_basis_biv, device=device, dtype=dtype))

        self._update_rotors()

    @staticmethod
    def _build_state(n, device, dtype):
        # Vectorized geom Cayley (avoids upstream's O(D^3) Python loop).
        from _cayley import build_geom_cayley
        cayley = build_geom_cayley(n, device=device).to(dtype=dtype)
        # ShortLex blade-degree vector — torch_ga's natural ShortLex order,
        # used by mv_reversion to apply per-grade signs.
        from torch_ga.cayley import blades_from_bases
        bases = [chr(ord("a") + i) for i in range(n)]
        _, blade_degrees_py = blades_from_bases(bases)
        blade_degrees = torch.tensor(blade_degrees_py, device=device)
        return {"cayley": cayley, "blade_degrees": blade_degrees}

    def _update_rotors(self):
        # Random even multivector: randn on even-grade slots, zero on odd.
        R = torch.randn(1, self.dim, device=self.device, dtype=self.dtype)
        R = R * (self._blade_degrees % 2 == 0).to(dtype=R.dtype)
        self._R     = R
        self._R_rev = mv_reversion(R, self._blade_degrees)

    def forward(self, x):
        # x: (batch, D) ShortLex, on `self.device`. Project sandwich: R~ x R.
        y = mv_multiply(self._R_rev, x, self._cayley)
        y = mv_multiply(y, self._R, self._cayley)
        return y


# ---------------------------------------------------------------------------
# Impl 2: Versor
# ---------------------------------------------------------------------------

class VersorRotor(nn.Module):
    """Rotor application via Versor: bivector → Versor.exp (general bivector,
    internal decomposition) → rotor multivector → Versor.sandwich_product
    (rebuilds an (N, D, D) action matrix per call; no precompute API).
    """
    _algebras = {}

    def __init__(self, dim, device='cuda', dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.n = int(math.log2(dim))
        self.num_basis_biv = math.comb(self.n, 2)

        key = (self.n, str(torch.device(device)), dtype)
        if VersorRotor._algebras.get(key) is None:
            VersorRotor._algebras[key] = VersorAlgebra(
                p=self.n, q=0, r=0, device=device, dtype=dtype)
        self.algebra = VersorRotor._algebras[key]

        self.bivectors_left = nn.Parameter(
            torch.randn(1, self.num_basis_biv, device=device, dtype=dtype))

        # bp_idx[k] = bit-pattern index of the k-th lex-pair bivector slot.
        # Used at every forward to scatter the (1, C(n,2)) parameter into a
        # (1, 2**n) bivector multivector in Versor's bit-pattern convention.
        bp_idx = _lex_pair_to_bp_indices(self.n).to(device)
        self.register_buffer('_bp_idx', bp_idx)

        self._update_rotors()

    def _bivector_mv(self):
        bp = torch.zeros(1, self.dim, device=self.device, dtype=self.dtype)
        bp[..., self._bp_idx] = self.bivectors_left
        return bp

    def _update_rotors(self):
        # Build the rotor multivector once. Versor.exp handles general bivectors.
        self._rotor = self.algebra.exp(self._bivector_mv())
        self._rotor_rev = self.algebra.reverse(self._rotor)

    def forward(self, x):
        # x: (batch, dim) in bit-pattern order.
        # Versor's per_channel_sandwich(R, x, R_rev) computes `R · x · R_rev`.
        # We pass R_rev as the "first rotor" and R as R_rev to apply `R~ · x · R`
        # (the project's sandwich convention).
        return self.algebra.per_channel_sandwich(
            self._rotor_rev, x.unsqueeze(1), R_rev=self._rotor,
        ).squeeze(1)


