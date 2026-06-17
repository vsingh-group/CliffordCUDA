"""wedge_prod (∧) — chunk + multik + subset_grade variants.

Public symbols come from the variant submodules:
  chunk.py        — chunk + multik kernels, with autograd
  subset_grade.py — subset-enumeration variant (forward-only)
"""
from .chunk import (  # noqa: F401
    _WedgeProdFunc, _WedgeProdMultikFunc,
    build_wedge_sign_bwd, build_wedge_sign_fwd,
    load_wedge_prod_cuda,
    wedge_prod, wedge_prod_skip,
    wedge_prod_multik, wedge_prod_multik_skip,
)
from .subset_grade import (  # noqa: F401
    build_wedge_subset_lut,
    load_wedge_prod_subset_grade_cuda,
    wedge_prod_subset_grade,
)
