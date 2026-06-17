"""Boundary validation helpers for CliffordAlgebra methods.

All checks raise `ValueError` with a clear message naming what was wrong so
failures surface at the call site, not deep inside a CUDA kernel TORCH_CHECK.
"""
from __future__ import annotations
import torch


def _check_device(device):
    """Resolve `device` to a concrete CUDA device, or raise at the boundary.

    cliffordcuda is CUDA-only (every kernel guards `is_cuda()`), so a non-CUDA
    device is rejected here rather than failing later inside a kernel. A bare
    `"cuda"` (no index) is resolved to the current device so the per-call
    device check compares a concrete index — closing the cross-GPU hole where
    a `cuda:1` tensor would pass against a `cuda:0`-built algebra."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(
            f"cliffordcuda is CUDA-only; device must be a CUDA device, got {dev}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cliffordcuda requires CUDA, but torch.cuda.is_available() is False")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _check_metric(metric):
    if not hasattr(metric, "__iter__"):
        raise ValueError("metric must be an iterable of {-1, 0, +1}")
    metric = tuple(int(m) for m in metric)
    if any(m not in (-1, 0, 1) for m in metric):
        raise ValueError(
            f"metric entries must each be in {{-1, 0, +1}}, got {metric}")
    if len(metric) < 1:
        raise ValueError("metric must have at least one entry")
    return metric


def _check_mv(tensor, name, dim, device, dtype):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2D (batch, 2^n); got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] != dim:
        raise ValueError(
            f"{name} last dim {tensor.shape[-1]} != 2^n = {dim}")
    if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index):
        raise ValueError(
            f"{name} on device {tensor.device}, algebra pinned to {device}")
    if tensor.dtype != dtype:
        raise ValueError(
            f"{name} dtype {tensor.dtype}, algebra dtype {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_bivector(b, n, num_basis_biv, device, dtype):
    if not isinstance(b, torch.Tensor):
        raise ValueError(f"bivector must be a torch.Tensor, got {type(b).__name__}")
    if b.dim() != 2:
        raise ValueError(
            f"bivector must be 2D (1, C(n, 2)); got shape {tuple(b.shape)}")
    if b.shape[-1] != num_basis_biv:
        raise ValueError(
            f"bivector last dim {b.shape[-1]} != C(n, 2) = {num_basis_biv} "
            f"(lex-pair order: e_01, e_02, ..., e_(n-2)(n-1))")
    if b.shape[0] != 1:
        raise ValueError(
            f"bivector first dim must be 1 (single rotor per call); "
            f"got shape {tuple(b.shape)}")
    if b.device.type != device.type or (
            device.index is not None and b.device.index != device.index):
        raise ValueError(
            f"bivector on device {b.device}, algebra pinned to {device}")
    if b.dtype != dtype:
        raise ValueError(
            f"bivector dtype {b.dtype}, algebra dtype {dtype}")
