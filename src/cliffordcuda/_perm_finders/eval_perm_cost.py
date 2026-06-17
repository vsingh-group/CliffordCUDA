"""Bank-conflict cost of a candidate permutation. Used by the GF(2) finder."""
import numpy as np

WARP = 32


def cost_of_perm(pi, ci_np, cj_np):
    """Total bank-conflict excess for permutation `pi` against the Givens
    rotation index arrays. Returns (total_cost, max_per_(rot,bank)_excess)."""
    num_rot, ppr = ci_np.shape
    target = ppr // WARP
    ci_perm = pi[ci_np] % WARP
    cj_perm = pi[cj_np] % WARP
    total = 0
    per_rot_max = 0
    for r in range(num_rot):
        for arr in (ci_perm[r], cj_perm[r]):
            counts = np.bincount(arr, minlength=WARP)
            excess = np.maximum(counts - target, 0)
            total += int(excess.sum())
            per_rot_max = max(per_rot_max, int(excess.max()))
    return total, per_rot_max
