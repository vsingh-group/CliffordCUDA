"""The public `CliffordAlgebra` class.

Bound at construction:
  * `metric` (length-n list/tuple of {-1, 0, +1}, defines Cl(p, q, r))
  * `n`, `dim = 2**n`, `device`, `dtype`

Per call:
  * GA products (`geom_prod`, `wedge_prod`, `inner_prod`, `left_contraction`,
    `right_contraction`, `regressive_prod`) take two `(batch, dim)`
    multivectors, return one.
  * `reverse(x)` returns the reversion of a multivector.
  * `dual(x)` returns the dual `x · I⁻¹` — a signed reversal of the
    coefficients (`compile=True` fuses the flip+scale via `torch.compile`).
  * `grade_involution(x)`, `clifford_conjugation(x)` — the other two
    grade-based sign involutions alongside `reverse`.
  * `grade_projection(x, k)` returns `<x>_k`: keeps grade k, zeros the rest.
  * `norm_sq(x)` returns the squared norm `<x x~>_0` (metric-weighted; one
    scalar per multivector; `compile=True` fuses the square-reduce).
  * Rotor application takes a bivector in lex-pair order
    `(1, C(n, 2))`. Two equivalent forms:
      - `apply_bivector(bivector, x)`: one shot, builds the Givens cs chain
        then applies. Use this every step in training.
      - `compile_bivector(bivector) -> cs`, `apply_rotor(cs, x) -> y`:
        precompute once for inference, then apply repeatedly without
        rebuilding cs.

No rotor multivector is ever materialized: the path is
`bivector → eigh-exp → Givens cs chain → kernel apply`.
"""
from __future__ import annotations
import math
import torch

from . import _validate
from ._utils.perm import build_permuted_indices
from ._utils.reorder import build_fused_rotation_indices_bank_optimized

from .extensions.ga.geom_prod import geom_prod as _geom_prod
from .extensions.ga.wedge_prod.chunk import wedge_prod as _wedge_prod
from .extensions.ga.inner_prod.chunk import inner_prod as _inner_prod
from .extensions.ga.left_contract import left_contract as _left_contract
from .extensions.ga.right_contract import right_contract as _right_contract
from .extensions.ga.regressive_prod import regressive_prod as _regressive_prod

from .extensions.rotor.eigh_exp import EighExpFunc
from .extensions.rotor.givens_factor import GivensFactorFunc
from .extensions.rotor.givens_apply.mb_perm_packed import (
    GivensApplyMbPermPackedFunc,
    load_givens_apply_mb_perm_packed_cuda,
    pack_indices as _pack_indices_perm,
)
from .extensions.rotor.givens_apply.mb_packed import (
    GivensApplyMbPackedFunc,
    load_givens_apply_mb_packed_cuda,
    pack_indices as _pack_indices,
)


_DUAL_COMPILED = None


def _dual_compiled():
    """The flip+scale of `dual`, fused into a single memory pass by
    `torch.compile`. Built once on first use (and shared across algebras,
    since it takes the sign vector as an argument) so that importing the
    library never pulls in a compiler backend."""
    global _DUAL_COMPILED
    if _DUAL_COMPILED is None:
        _DUAL_COMPILED = torch.compile(lambda x, signs: x.flip(-1) * signs)
    return _DUAL_COMPILED


_NORM_SQ_COMPILED = None


def _norm_sq_compiled():
    """The square-weight-reduce of `norm_sq`, fused into a single
    read-and-reduce by `torch.compile`. Built once on first use and shared
    across algebras (it takes the weight vector as an argument), so importing
    the library never pulls in a compiler backend."""
    global _NORM_SQ_COMPILED
    if _NORM_SQ_COMPILED is None:
        _NORM_SQ_COMPILED = torch.compile(lambda x, w: (x * x * w).sum(-1))
    return _NORM_SQ_COMPILED


class CliffordAlgebra:
    def __init__(self, metric, device="cuda"):
        metric = _validate._check_metric(metric)
        self.metric = metric
        self.n = len(metric)
        self.dim = 1 << self.n
        self.device = _validate._check_device(device)
        # The CUDA kernels are fp32-only, so the algebra is always fp32. Kept
        # as an attribute because the boundary checks and table builders read
        # it, but it is not a constructor knob (there is no other valid value).
        self.dtype = torch.float32

        self._num_basis_biv = self.n * (self.n - 1) // 2

        # Rotor-apply kernel variant selection by n. At n >= 9 the GF(2)
        # permutation is feasible (drop-2 rank-5 constraint); for 7 <= n < 9
        # the CP-SAT-found bank-permuted reorder kernel is used. K is chosen
        # so ppr / K is a warp multiple. ppr = 2^(n-2) needs to be at least
        # 32 (one full warp), so the rotor path is supported for n >= 7.
        self._rotor_supported = self.n >= 7
        if self._rotor_supported:
            ppr = 1 << (self.n - 2)
            self._rotor_K = min(8, max(1, ppr // 32))
            self._rotor_M = 2
            self._rotor_variant = "gf2" if self.n >= 9 else "cpsat"
            self._init_rotor_buffers()

    # ── Rotor-apply construction ─────────────────────────────────────
    def _init_rotor_buffers(self):
        """Precompute the per-n structural data the rotor kernels need.

        The rotor kernels and the rotation tables are now bit-pattern
        native (see `_utils/core.build_fused_rotation_indices_bp`), so the
        kernel consumes bit-pattern multivectors directly — no bp<->sl
        translation around the call.
        """
        device = self.device
        dtype  = self.dtype

        skew_i, skew_j = torch.triu_indices(self.n, self.n, offset=1, device=device)
        self._skew_i = skew_i
        self._skew_j = skew_j

        # Both variants apply a bank-conflict-minimising permutation, then the
        # same permuted Givens-apply kernel. The variant only selects which
        # finder produced the permutation: CP-SAT for 7 <= n < 9, GF(2) for
        # n >= 9 (see _rotor_variant above).
        perm, ci, cj, csig = build_permuted_indices(
            self.n, dtype, device, variant=self._rotor_variant)
        packed_ij, packed_sig = _pack_indices_perm(ci, cj, csig)
        self._perm = perm
        self._packed_ij = packed_ij
        self._packed_sig = packed_sig

    # ── GA products ───────────────────────────────────────────────────
    def geom_prod(self, a, b):
        _validate._check_mv(a, "a", self.dim, self.device, self.dtype)
        _validate._check_mv(b, "b", self.dim, self.device, self.dtype)
        return _geom_prod(a, b, metric=self.metric)

    def wedge_prod(self, a, b):
        _validate._check_mv(a, "a", self.dim, self.device, self.dtype)
        _validate._check_mv(b, "b", self.dim, self.device, self.dtype)
        return _wedge_prod(a, b, metric=self.metric)

    def inner_prod(self, a, b):
        _validate._check_mv(a, "a", self.dim, self.device, self.dtype)
        _validate._check_mv(b, "b", self.dim, self.device, self.dtype)
        return _inner_prod(a, b, metric=self.metric)

    def left_contraction(self, a, b):
        _validate._check_mv(a, "a", self.dim, self.device, self.dtype)
        _validate._check_mv(b, "b", self.dim, self.device, self.dtype)
        return _left_contract(a, b, metric=self.metric)

    def right_contraction(self, a, b):
        _validate._check_mv(a, "a", self.dim, self.device, self.dtype)
        _validate._check_mv(b, "b", self.dim, self.device, self.dtype)
        return _right_contract(a, b, metric=self.metric)

    def regressive_prod(self, a, b):
        if 0 in self.metric:
            raise ValueError(
                "regressive_prod is undefined for degenerate metrics "
                "(any 0 in the metric makes the pseudoscalar non-invertible).")
        _validate._check_mv(a, "a", self.dim, self.device, self.dtype)
        _validate._check_mv(b, "b", self.dim, self.device, self.dtype)
        return _regressive_prod(a, b, metric=self.metric)

    # ── Reversion ─────────────────────────────────────────────────────
    def reverse(self, x):
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        if not hasattr(self, "_reverse_signs"):
            # Pre-compute on first call.
            idx = torch.arange(self.dim, device=self.device)
            k = torch.zeros(self.dim, device=self.device, dtype=torch.long)
            for b in range(self.n):
                k += (idx >> b) & 1
            sig = torch.where(((k * (k - 1) // 2) % 2) == 1, -1.0, 1.0)
            self._reverse_signs = sig.to(self.dtype)
        return x * self._reverse_signs

    # ── Dual ──────────────────────────────────────────────────────────
    def dual(self, x, compile=False):
        """Dual `x* = x · I⁻¹` (right product with the inverse unit
        pseudoscalar `I = e_1…e_n`).

        In an orthogonal basis this is a signed permutation of the 2**n
        coefficients: complementing the blade bitmask sends grade k to grade
        n-k, which in bit-pattern order is just a reversal of the last axis,
        and each coefficient picks up a ±1. So the whole op is
        `x.flip(-1) * signs`, with `signs` precomputed once per (n, metric).
        It is memory-bound; there is no kernel.

        `I⁻¹` exists only for a non-degenerate metric, so a metric containing
        0 is rejected (same restriction as `regressive_prod`).

        compile: if True, run the flip+scale through a cached `torch.compile`d
            kernel that fuses them into one memory pass (hits the
            memory-bandwidth floor; ~2-3x over eager at large batch). Leave it
            False when the surrounding model is itself compiled — the eager
            path traces with no graph breaks and gets fused there for free.
        """
        if 0 in self.metric:
            raise ValueError(
                "dual is undefined for degenerate metrics (any 0 in the "
                "metric makes the pseudoscalar I non-invertible).")
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        if not hasattr(self, "_dual_signs"):
            self._dual_signs = self._build_dual_signs()
        if compile:
            return _dual_compiled()(x, self._dual_signs)
        return x.flip(-1) * self._dual_signs

    def _build_dual_signs(self):
        """Per-output-position ±1 for `x ↦ x · I⁻¹`. Output position m draws
        from input blade `a = (dim-1) ^ m` (the bit-complement); the factor is
        the geometric-product sign of `e_a · I⁻¹`:

            (1 / I²) · (-1)^(Σ set-bit-positions of a) · ∏_{i∈a} metric[i]

        with `I² = (-1)^(n(n-1)/2) · ∏ metric` (±1 for a non-degenerate
        metric, so 1/I² == I²). The reorder term counts the swaps to move e_a
        past the full pseudoscalar; the metric term contracts the shared
        indices (all of a, since a ⊆ full)."""
        n, D, full = self.n, self.dim, self.dim - 1
        i_sq = (-1) ** (n * (n - 1) // 2)
        for m in self.metric:
            i_sq *= m
        inv_i_sq = float(i_sq)          # ±1, so 1/I² == I²
        idx = torch.arange(D, device=self.device)
        a = full ^ idx                  # input blade feeding output position idx
        swaps = torch.zeros(D, device=self.device, dtype=torch.long)
        metric_prod = torch.ones(D, device=self.device)
        for i in range(n):
            bit = (a >> i) & 1
            swaps += i * bit
            metric_prod = metric_prod * torch.where(
                bit.bool(), float(self.metric[i]), 1.0)
        reorder = torch.where((swaps % 2) == 1, -1.0, 1.0)
        return (inv_i_sq * reorder * metric_prod).to(self.dtype)

    # ── Grade-based involutions and projection ────────────────────────
    def _grade_of_each_blade(self):
        """Grade (popcount) of every bit-pattern blade index, cached."""
        if not hasattr(self, "_grades"):
            idx = torch.arange(self.dim, device=self.device)
            k = torch.zeros(self.dim, device=self.device, dtype=torch.long)
            for b in range(self.n):
                k += (idx >> b) & 1
            self._grades = k
        return self._grades

    def grade_involution(self, x):
        """Grade involution `x^`: negate the odd-grade parts
        (grade k -> (-1)^k). An automorphism."""
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        if not hasattr(self, "_grade_involution_signs"):
            k = self._grade_of_each_blade()
            self._grade_involution_signs = torch.where(
                (k % 2) == 1, -1.0, 1.0).to(self.dtype)
        return x * self._grade_involution_signs

    def clifford_conjugation(self, x):
        """Clifford conjugation `x-`: grade k -> (-1)^(k(k+1)/2). Equals
        reverse composed with grade involution; an anti-automorphism."""
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        if not hasattr(self, "_clifford_conj_signs"):
            k = self._grade_of_each_blade()
            self._clifford_conj_signs = torch.where(
                ((k * (k + 1) // 2) % 2) == 1, -1.0, 1.0).to(self.dtype)
        return x * self._clifford_conj_signs

    def grade_projection(self, x, grade):
        """Grade projection `<x>_grade`: keep the grade-`grade` coefficients
        and zero out all others. Output keeps the full (batch, 2**n) shape."""
        if not 0 <= grade <= self.n:
            raise ValueError(
                f"grade must be in [0, {self.n}] for n={self.n}; got {grade}")
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        mask = (self._grade_of_each_blade() == grade).to(self.dtype)
        return x * mask

    # ── Norm ──────────────────────────────────────────────────────────
    def norm_sq(self, x, compile=False):
        """Squared norm `<x x~>_0 = sum_b x_b^2 * prod_{i in b} metric[i]`,
        returned as one scalar per multivector (shape `(batch,)`).

        The metric enters only through the per-blade weight `prod metric[i]`:
        all +1 for a Euclidean metric (so this is `sum_b x_b^2`), ±1 for an
        indefinite metric (so the result may be negative), and 0 on any blade
        containing a null generator for a degenerate metric.

        compile: if True, run the square-weight-reduce through a cached
            `torch.compile`d kernel that fuses it into a single read-and-reduce
            (hits the memory-bandwidth floor; ~5x over eager at large batch,
            which makes three passes with intermediates). Leave it False when
            the surrounding model is itself compiled — the eager path fuses
            there for free."""
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        if not hasattr(self, "_norm_weights"):
            idx = torch.arange(self.dim, device=self.device)
            w = torch.ones(self.dim, device=self.device)
            for i in range(self.n):
                bit = (idx >> i) & 1
                w = w * torch.where(bit.bool(), float(self.metric[i]), 1.0)
            self._norm_weights = w.to(self.dtype)
        if compile:
            return _norm_sq_compiled()(x, self._norm_weights)
        return (x * x * self._norm_weights).sum(-1)

    # ── Rotor application ────────────────────────────────────────────
    def _check_rotor_supported(self):
        if not self._rotor_supported:
            raise ValueError(
                f"rotor application requires n >= 7 (ppr = 2^(n-2) must be "
                f"at least 32 = warp size); got n={self.n}")

    def compile_bivector(self, bivector):
        """Build the Givens cs chain for the given bivector. Returns a tensor
        that `apply_rotor` consumes."""
        self._check_rotor_supported()
        _validate._check_bivector(bivector, self.n, self._num_basis_biv,
                                  self.device, self.dtype)
        R = EighExpFunc.apply(
            bivector, self._skew_i, self._skew_j, self.n, self.n // 2)
        cs = GivensFactorFunc.apply(R)[0]
        return cs

    def apply_rotor(self, cs, x):
        """Apply a precompiled rotor (cs from `compile_bivector`) to a batch
        of bit-pattern multivectors. Kernel is bit-pattern native — no
        translation around the call."""
        self._check_rotor_supported()
        _validate._check_mv(x, "x", self.dim, self.device, self.dtype)
        # Both variants (gf2, cpsat) use the permuted Givens-apply kernel.
        if torch.is_grad_enabled():
            return GivensApplyMbPermPackedFunc.apply(
                x, cs, self._packed_ij, self._packed_sig, self._perm,
                self._rotor_K, self._rotor_M)
        mod = load_givens_apply_mb_perm_packed_cuda()
        return mod.givens_mb_perm_packed_fwd(
            x.contiguous(), cs.contiguous(),
            self._packed_ij, self._packed_sig, self._perm,
            self._rotor_K, self._rotor_M)

    def apply_bivector(self, bivector, x):
        """Convenience: `apply_rotor(compile_bivector(bivector), x)`."""
        return self.apply_rotor(self.compile_bivector(bivector), x)

    # ── Device move (returns a new instance bound to the new device) ─
    def to(self, device):
        return CliffordAlgebra(self.metric, device=device)

    def __repr__(self):
        p = sum(1 for m in self.metric if m == 1)
        q = sum(1 for m in self.metric if m == -1)
        r = sum(1 for m in self.metric if m == 0)
        return (f"CliffordAlgebra(metric={list(self.metric)}, "
                f"Cl({p}, {q}, {r}), n={self.n}, dim={self.dim}, "
                f"device={self.device})")
