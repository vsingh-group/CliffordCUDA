"""Givens-apply rotor kernels (multi-batch packed + permuted variants)."""
import torch


def pack_indices(ci, cj, csig):
    """Pack (ci, cj) into one int32 (16 bits each) and csig into a sign bitmask.

    Shared by the mb_packed and mb_perm_packed variants (the packing is
    identical; only the kernel that consumes it differs)."""
    packed_ij = (ci.to(torch.int32) << 16) | cj.to(torch.int32)
    num_rot, ppr = csig.shape
    assert ppr % 32 == 0, "ppr must be divisible by 32 for sign bitmask"
    sig_neg = (csig < 0).to(torch.int32).view(num_rot, ppr // 32, 32)
    bit_positions = torch.arange(32, device=sig_neg.device, dtype=torch.int32)
    packed_sig = (sig_neg << bit_positions).sum(dim=-1, dtype=torch.int32)
    return packed_ij.contiguous(), packed_sig.contiguous()
