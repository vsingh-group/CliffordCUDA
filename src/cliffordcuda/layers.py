"""``nn.Module`` layers wrapping the CliffordAlgebra products and rotor.

Each layer holds a learnable parameter and applies one operation against the
batched input ``x`` of shape ``(batch, 2^n)`` (bit-pattern blade order, fp32,
on the algebra's CUDA device).

  - The product layers (geometric, wedge, inner, left/right contraction,
    regressive) hold a learnable multivector ``weight`` of shape ``(1, 2^n)``
    and compute ``product(x, weight)`` — input on the left, weight on the
    right (the products are non-commutative, so order matters). The weight is
    broadcast over the batch, so its gradient accumulates across batch
    elements.
  - ``RotorLayer`` holds a learnable ``bivector`` of shape ``(1, C(n, 2))``
    and computes the sandwich ``R~ x R`` via ``apply_bivector``. The rotor is
    rebuilt from the current bivector on every forward (so the bivector is
    what's learned). Requires ``n >= 7``.

All layers take an existing ``CliffordAlgebra`` so several layers can share
one algebra (and its lookup tables / device). Example::

    cl = CliffordAlgebra(metric=[1, 1, 1, 1, -1], device="cuda")
    layer = GeometricProductLayer(cl)
    y = layer(x)          # x: (batch, 32) fp32 cuda; y: same shape
    y.sum().backward()    # gradient flows to layer.weight
"""
import torch
import torch.nn as nn


class _ProductLayer(nn.Module):
    """Base for the binary-product layers: a learnable ``(1, 2^n)`` multivector
    weight, applied as ``product(x, weight)``. Subclasses set ``_method`` to the
    name of the CliffordAlgebra method to call."""

    _method: str = None

    def __init__(self, cl):
        super().__init__()
        self.cl = cl
        self.weight = nn.Parameter(
            torch.randn(1, cl.dim, device=cl.device, dtype=cl.dtype))

    def forward(self, x):
        # Broadcast the shared weight to the batch; the products require both
        # operands to be (batch, 2^n) and contiguous.
        w = self.weight.expand_as(x).contiguous()
        return getattr(self.cl, self._method)(x, w)

    def extra_repr(self):
        return f"n={self.cl.n}, dim={self.cl.dim}, op={self._method}"


class GeometricProductLayer(_ProductLayer):
    _method = "geom_prod"


class WedgeProductLayer(_ProductLayer):
    _method = "wedge_prod"


class InnerProductLayer(_ProductLayer):
    _method = "inner_prod"


class LeftContractionLayer(_ProductLayer):
    _method = "left_contraction"


class RightContractionLayer(_ProductLayer):
    _method = "right_contraction"


class RegressiveProductLayer(_ProductLayer):
    # Defined for non-degenerate signatures only; raises on r > 0 (see
    # CliffordAlgebra.regressive_prod).
    _method = "regressive_prod"


class RotorLayer(nn.Module):
    """Learnable rotor: a (1, C(n, 2)) bivector parameter, applied as the
    sandwich R~ x R. Requires n >= 7. Recompiled each forward in train();
    compiled once and cached in eval()."""

    def __init__(self, cl):
        super().__init__()
        if cl.n < 7:
            raise ValueError(
                f"RotorLayer requires n>=7 (rotor support); got n={cl.n}")
        self.cl = cl
        num_basis_biv = cl.n * (cl.n - 1) // 2
        self.bivector = nn.Parameter(
            torch.randn(1, num_basis_biv, device=cl.device, dtype=cl.dtype))
        self._cs = None

    def train(self, mode=True):
        # Leaving eval (or re-entering train) drops the inference rotor cache.
        self._cs = None
        return super().train(mode)

    def forward(self, x):
        if self.training:
            return self.cl.apply_bivector(self.bivector, x)
        if self._cs is None:                       # eval: compile once, reuse
            self._cs = self.cl.compile_bivector(self.bivector)
        return self.cl.apply_rotor(self._cs, x)

    def extra_repr(self):
        return f"n={self.cl.n}, dim={self.cl.dim}, num_basis_biv={self.bivector.shape[-1]}"
