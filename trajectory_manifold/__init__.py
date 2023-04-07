"""A Python library for computing statistics on trajectory manifolds."""

__all__ = ["SolverParameters",
           "system_pushforward_weight",
           "system_sensitivity",
           "trapezoidal_correlation",
           "trapezoidal_inner_product",
           "frobenius_inner_product"]

from .main import trapezoidal_inner_product
from .main import system_pushforward_weight
from .main import system_sensitivity
from .main import trapezoidal_correlation
from .main import frobenius_inner_product
from .main import SolverParameters