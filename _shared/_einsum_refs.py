"""Einsum-style reference implementations for the GA products.

Pattern shared by every product:

    outer  = einsum("...i,...j->...ij", A, B)          # (..., D, D)
    signed = outer * signs                              # (D, D) sign+mask table
    out[..., k] = scatter_add(signed[..., i, j], k = i XOR j)

The (D, D) `signs` table is the metric-aware geometric-product reorder sign
times a per-product mask:
  geom_prod     : no mask
  wedge_prod    : (i & j) == 0                  (disjoint generators)
  inner_prod    : (i sub j) or (j sub i)        (Hestenes inner)
  left_contract : i sub j                       ((i & j) == i)
  right_contract: j sub i                       ((i & j) == j)

The factored representation uses two (D, D) tables instead of a dense
(D, D, D) Cayley — same compute, much less memory at high n.
"""
from cliffordcuda.extensions.ga.geom_prod import (
    _build_sign_value_table, _normalize_metric,
)
import torch


class _EinsumBase:
    """Builds the (D, D) sign-and-mask table once; subclasses just set `mask`."""

    def __init__(self, n: int, metric=None, device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        self.n = n
        self.dim = 1 << n
        self.device = device
        self.dtype = dtype

        i = torch.arange(self.dim, dtype=torch.long, device=device)
        I = i.view(-1, 1).expand(self.dim, self.dim)
        J = i.view(1, -1).expand(self.dim, self.dim)
        self.indices = (I ^ J).contiguous().to(torch.long)

        metric = _normalize_metric(n, metric)
        sigma = _build_sign_value_table(n, metric).to(device=device, dtype=dtype)
        self.signs = sigma * self._build_mask(I, J).to(dtype)

    def _build_mask(self, I, J):
        return torch.ones_like(I, dtype=torch.bool)

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        outer = torch.einsum("...i,...j->...ij", A, B)
        signed = outer * self.signs
        flat = signed.reshape(*signed.shape[:-2], -1)
        flat_idx = self.indices.reshape(-1).expand_as(flat)
        out = torch.zeros(*flat.shape[:-1], self.dim,
                          dtype=A.dtype, device=A.device)
        return out.scatter_add(-1, flat_idx, flat)


class EinsumGP(_EinsumBase):
    """Geometric product: no mask."""


class EinsumWedge(_EinsumBase):
    """Wedge: disjoint generators."""
    def _build_mask(self, I, J):
        return (I & J) == 0


class EinsumInner(_EinsumBase):
    """Hestenes inner: i subset j OR j subset i."""
    def _build_mask(self, I, J):
        return ((I & (~J)) == 0) | ((J & (~I)) == 0)


class EinsumLeftContract(_EinsumBase):
    """Left contraction: i subset j."""
    def _build_mask(self, I, J):
        return (I & J) == I


class EinsumRightContract(_EinsumBase):
    """Right contraction: j subset i."""
    def _build_mask(self, I, J):
        return (I & J) == J


class EinsumRegressive:
    """Regressive (meet) product via the metric-aware definition

        a v b = dual( dual(a) /\\ dual(b) )

    where /\\ is the einsum exterior product and `dual(e_J)` is computed from
    the metric's pseudoscalar:

        dual(e_J) = e_J . I^{-1}
                  = (1/I^2) * sigma[J, full] * e_{J xor full}

    Requires a non-degenerate metric (no zero generators) — I^2 = 0 otherwise.
    """

    def __init__(self, n: int, metric=None, device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        self.n = n
        self.dim = 1 << n
        metric = _normalize_metric(n, metric)
        if 0 in metric:
            raise ValueError(
                f"regressive product undefined for degenerate metric {metric}"
            )
        sigma = _build_sign_value_table(n, metric).to(device=device, dtype=dtype)
        full = self.dim - 1
        Isq = float(sigma[full, full].item())   # +/- 1 for non-degenerate
        # Permutation: result blade k corresponds to input blade `comp[k] = ~k`.
        comp = torch.arange(self.dim, device=device, dtype=torch.long) ^ full
        # dual(x)[k] = x[comp[k]] * dual_sign[k]
        #            = x[~k] * (1/I^2) * sigma[~k, full]
        self._comp = comp
        self._dual_sign = (sigma[comp, full] / Isq).contiguous()
        self._wedge = EinsumWedge(n, metric=metric, device=device, dtype=dtype)

    def _dual(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self._comp) * self._dual_sign

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self._dual(self._wedge(self._dual(A), self._dual(B)))
