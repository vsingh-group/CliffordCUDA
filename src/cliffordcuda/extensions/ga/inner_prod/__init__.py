"""inner_prod (Hestenes) — chunk + multik + subset_grade variants.

Public symbols come from the variant submodules:
  chunk.py        — chunk + multik kernels, with autograd
  subset_grade.py — subset-enumeration variant (forward-only)
"""
from .chunk import (  # noqa: F401
    _InnerProdFunc, _InnerProdMultikFunc,
    build_inner_sign_bwd, build_inner_sign_fwd,
    load_inner_prod_cuda,
    inner_prod, inner_prod_skip, inner_prod_kskip,
    inner_prod_multik, inner_prod_multik_skip,
)
from .subset_grade import (  # noqa: F401
    build_inner_subset_lut,
    inner_prod_subset_grade,
    load_inner_prod_subset_grade_cuda,
    _normalize_metric, _sigma_val,
)
