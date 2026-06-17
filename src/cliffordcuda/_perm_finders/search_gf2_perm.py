"""GF(2)-linear permutation search for the bank-conflict-free Givens apply kernel.

Importable as `from cliffordcuda._perm_finders import search_gf2_perm`;
`find(n, warp_size, time_limit_sec)` returns the permutation as a numpy array.
"""
import argparse
import time
from itertools import combinations

import numpy as np
import torch

from .._utils.core import build_fused_rotation_indices_bp
from .eval_perm_cost import cost_of_perm


BANK_BITS = 5


def _identity_bitpatterns(n):
    """For bit-pattern tables we don't permute the blade space — each blade
    index already IS its own bit-pattern. Returned for the _perm_from_M
    helper which composes the linear map M with the blade bit decomposition."""
    return np.arange(1 << n, dtype=np.int64)


def _rank_gf2(M):
    M = M.copy().astype(np.uint8)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r >= rows: break
        pivot = None
        for i in range(r, rows):
            if M[i, c]:
                pivot = i; break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def _all_drop2_rank5(A):
    n_cols = A.shape[1]
    for p in range(n_cols):
        for q in range(p + 1, n_cols):
            cols = [c for c in range(n_cols) if c != p and c != q]
            if _rank_gf2(A[:, cols]) < BANK_BITS:
                return False
    return True


def _perm_from_M(M, bit_patterns):
    n = M.shape[1]
    dim = 2 ** n
    out = np.zeros(dim, dtype=np.int64)
    for r in range(M.shape[0]):
        bit = np.zeros(dim, dtype=np.int64)
        for c in range(n):
            if M[r, c]:
                bit ^= ((bit_patterns >> c) & 1)
        out |= (bit << r)
    return out


def find(n: int, warp_size: int = 32, time_limit_sec: int = 60) -> np.ndarray:
    """Return a GF(2)-linear bank-conflict-friendly permutation of [0, 2^n).
    Feasible for n >= 9 (the drop-2 rank-5 constraint is too tight at lower n)."""
    assert warp_size == 32, "GF(2) search hardcoded to warp_size=32"
    ci, cj, _ = build_fused_rotation_indices_bp(n, torch.float32, "cpu")
    ci_np, cj_np = ci.numpy().astype(np.int64), cj.numpy().astype(np.int64)
    dim = 2 ** n
    bit_patterns = _identity_bitpatterns(n)

    best = None
    t0 = time.time()
    rng = np.random.default_rng(0)
    n_tries = 0
    while time.time() - t0 < time_limit_sec:
        Mfull = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        if _rank_gf2(Mfull) < n:
            n_tries += 1; continue
        A_top = Mfull[:BANK_BITS]
        if not _all_drop2_rank5(A_top):
            n_tries += 1; continue
        pi = _perm_from_M(Mfull, bit_patterns)
        if len(np.unique(pi)) != dim:
            n_tries += 1; continue
        c, mx = cost_of_perm(pi, ci_np, cj_np)
        if best is None or c < best[0]:
            best = (c, mx, pi)
            print(f"  gf2: [t={time.time()-t0:.1f}s try={n_tries}] cost={c}", flush=True)
            if c == 0:
                break
        n_tries += 1

    if best is None:
        raise RuntimeError(
            f"gf2: no feasible permutation for n={n} within {time_limit_sec}s "
            f"(drop-2 rank-{BANK_BITS} infeasible? try n >= 9)")
    return best[2]


def _cli():
    """`python -m cliffordcuda._perm_finders.search_gf2_perm --n N`"""
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--time", type=int, default=60)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    pi = find(args.n, warp_size=32, time_limit_sec=args.time)
    if args.out:
        torch.save(torch.from_numpy(pi), args.out)
        print(f"saved {args.out}")
    else:
        print(pi)


if __name__ == "__main__":
    _cli()
