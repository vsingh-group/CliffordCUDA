"""cliffordcuda: custom CUDA kernels for Cl(p, q, r) geometric algebra."""
from .algebra import CliffordAlgebra
from .extensions.ga.geom_prod import geom_prod, geom_prod_multik
from .extensions.ga.wedge_prod import (
    wedge_prod, wedge_prod_multik, wedge_prod_skip,
    wedge_prod_multik_skip, wedge_prod_subset_grade,
)
from .extensions.ga.inner_prod import (
    inner_prod, inner_prod_kskip, inner_prod_multik,
    inner_prod_multik_skip, inner_prod_skip, inner_prod_subset_grade,
)
from .extensions.ga.left_contract import (
    left_contract, left_contract_skip, left_contract_subset_grade,
)
from .extensions.ga.right_contract import (
    right_contract, right_contract_skip, right_contract_subset_grade,
)
from .extensions.ga.regressive_prod import (
    regressive_prod, regressive_prod_skip, regressive_prod_subset_grade,
)
from .layers import (
    GeometricProductLayer, WedgeProductLayer, InnerProductLayer,
    LeftContractionLayer, RightContractionLayer, RegressiveProductLayer,
    RotorLayer,
)

__all__ = [
    "CliffordAlgebra",
    "geom_prod", "geom_prod_multik",
    "wedge_prod", "wedge_prod_multik", "wedge_prod_skip",
    "wedge_prod_multik_skip", "wedge_prod_subset_grade",
    "inner_prod", "inner_prod_kskip", "inner_prod_multik",
    "inner_prod_multik_skip", "inner_prod_skip", "inner_prod_subset_grade",
    "left_contract", "left_contract_skip", "left_contract_subset_grade",
    "right_contract", "right_contract_skip", "right_contract_subset_grade",
    "regressive_prod", "regressive_prod_skip", "regressive_prod_subset_grade",
    "GeometricProductLayer", "WedgeProductLayer", "InnerProductLayer",
    "LeftContractionLayer", "RightContractionLayer", "RegressiveProductLayer",
    "RotorLayer",
]
__version__ = "0.1.0"
