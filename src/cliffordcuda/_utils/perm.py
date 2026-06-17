"""Find and load the bank-conflict-free permutation for the Givens apply kernel.

Resolution order for `n{n}_w{warp_size}{_variant}.pt`:
  1. shipped read-only cache at `cliffordcuda/_data/perm/`
  2. user XDG cache at `~/.cache/cliffordcuda/perm/`
  3. compute via the finder, save to (2) for next time

Missing perm files are NOT an error — the class transparently regenerates
them, with a one-line notice the first time (the search can take real
wall time at higher n).
"""
import os
import numpy as np
import torch
from .core import build_fused_rotation_indices_bp
from .reorder import _bipartite_chunk_assignment
from .._config import _DATA_DIR, _USER_CACHE_DIR


_SHIPPED_CACHE_DIR = _DATA_DIR / "perm"
_USER_PERM_CACHE_DIR = _USER_CACHE_DIR / "perm"


def _cache_filename(n: int, warp_size: int, variant: str) -> str:
    suffix = "" if variant == "cpsat" else f"_{variant}"
    return f"n{n}_w{warp_size}{suffix}_bp.pt"


def _load_perm(n: int, warp_size: int = 32, variant: str = "cpsat") -> np.ndarray:
    fname = _cache_filename(n, warp_size, variant)
    shipped = _SHIPPED_CACHE_DIR / fname
    user    = _USER_PERM_CACHE_DIR / fname

    for path in (shipped, user):
        if path.exists():
            return torch.load(str(path), map_location="cpu", weights_only=True).numpy()

    # No cached perm — compute, save to user cache.
    print(
        f"  cliffordcuda: no cached {variant} permutation for n={n}; "
        f"computing (this can take a few minutes at high n) ...",
        flush=True,
    )
    if variant == "gf2":
        from .._perm_finders import search_gf2_perm as finder
    elif variant == "cpsat":
        from .._perm_finders import permute_search_cpsat as finder
    else:
        raise ValueError(f"unknown permutation variant: {variant!r}")

    pi = finder.find(n=n, warp_size=warp_size)   # finder returns np.ndarray
    _USER_PERM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(pi), str(user))
    print(f"  cliffordcuda: wrote {user}", flush=True)
    return pi


def build_permuted_indices(n, dtype, device, warp_size=32, variant="cpsat"):
    pi = _load_perm(n, warp_size=warp_size, variant=variant)

    ci_raw, cj_raw, csig_raw = build_fused_rotation_indices_bp(n, dtype, device)

    pi_t = torch.from_numpy(pi).to(device).long()
    ci_perm = pi_t[ci_raw.long()].to(torch.int32).contiguous()
    cj_perm = pi_t[cj_raw.long()].to(torch.int32).contiguous()

    ci_cpu = ci_perm.cpu().tolist()
    cj_cpu = cj_perm.cpu().tolist()
    csig_cpu = csig_raw.cpu().tolist()
    num_rot, ppr = ci_perm.shape

    new_ci = [[0] * ppr for _ in range(num_rot)]
    new_cj = [[0] * ppr for _ in range(num_rot)]
    new_csig = [[0.0] * ppr for _ in range(num_rot)]
    for r in range(num_rot):
        ci_banks = [v % warp_size for v in ci_cpu[r]]
        cj_banks = [v % warp_size for v in cj_cpu[r]]
        chunks = _bipartite_chunk_assignment(ci_banks, cj_banks, warp_size)
        order = []
        for chunk in chunks:
            order.extend(chunk)
        for new_p, old_p in enumerate(order):
            new_ci[r][new_p] = ci_cpu[r][old_p]
            new_cj[r][new_p] = cj_cpu[r][old_p]
            new_csig[r][new_p] = csig_cpu[r][old_p]

    perm_t = torch.from_numpy(pi.astype(np.int32)).to(device).contiguous()
    ci_t = torch.tensor(new_ci, dtype=torch.int32, device=device).contiguous()
    cj_t = torch.tensor(new_cj, dtype=torch.int32, device=device).contiguous()
    csig_t = torch.tensor(new_csig, dtype=dtype, device=device).contiguous()
    return perm_t, ci_t, cj_t, csig_t
