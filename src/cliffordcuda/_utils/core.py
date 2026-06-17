import torch
import math


def build_fused_rotation_indices_bp(n, dtype, device):
    """Per-plane Givens rotation index/sign tables in bit-pattern order.

    For each rotation plane (p, q) with p < q, enumerates every bit-pattern
    index `i` with `bit_p(i) = 1` and `bit_q(i) = 0`; pairs with
    `j = i XOR ((1<<p)|(1<<q))`. Sign uses the blade-reorder parity formula in
    bit-pattern terms:

      removal_swaps = popcount(i) - 1 - popcount(i & ((1<<p) - 1))
                      (position of bit p in the sorted basis of i)
      inv_insert    = popcount(i >> (q + 1))
                      (basis bits in i above q, which q hops over when
                       inserted in sorted order)

    Rotation enumeration order is reversed `(p, q)` lex, so plane (p, q) sits
    at a fixed `rot_idx`.
    """
    num_rot = math.comb(n, 2)
    pairs_per_rot = 2 ** (n - 2)
    dim = 1 << n
    pq_reversed = list(reversed(
        [(p, q) for p in range(n - 1) for q in range(p + 1, n)]))

    ci_t   = torch.zeros(num_rot, pairs_per_rot, dtype=torch.int32, device=device)
    cj_t   = torch.zeros(num_rot, pairs_per_rot, dtype=torch.int32, device=device)
    csig_t = torch.zeros(num_rot, pairs_per_rot, dtype=dtype,        device=device)

    for rot_idx, (p, q) in enumerate(pq_reversed):
        mask = (1 << p) | (1 << q)
        slot = 0
        for i in range(dim):
            if ((i >> p) & 1) == 1 and ((i >> q) & 1) == 0:
                j = i ^ mask
                pc_total   = bin(i).count("1")
                pc_below_p = bin(i & ((1 << p) - 1)).count("1")
                removal_swaps = pc_total - 1 - pc_below_p
                inv_insert    = bin(i >> (q + 1)).count("1")
                sign = -1.0 if ((removal_swaps + inv_insert) & 1) else 1.0
                ci_t[rot_idx, slot] = i
                cj_t[rot_idx, slot] = j
                csig_t[rot_idx, slot] = sign
                slot += 1
        assert slot == pairs_per_rot, \
            f"chunk size mismatch at (p, q) = ({p}, {q}): got {slot}"

    return ci_t.contiguous(), cj_t.contiguous(), csig_t.contiguous()
