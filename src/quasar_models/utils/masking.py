"""
This file contains utilities for efficiently calculating coverage masks, i.e.
the ranges of arrays that fall within the bounds [lb, ub] (both inclusive). 

These functions assume the following:
1. The input arrays are sorted and purely finite. 
2. The input arrays are on a grid, e.g. linearly or logarithmicaly spaced, the 
   latter being more relevant to this project.
"""
from math import ceil, floor, log
from numpy import float64, bool_, zeros
from numpy.typing import NDArray

def linear_get_lower_index(
    x: NDArray[float64],
    lb: float,
    *,
    dx: float | None = None,
) -> int:
    """
    Returns the integer, 'i', s.t. 'x[i] >= lb'.
    """
    if not dx:
        dx = x[1] - x[0]
    return ceil((lb - x[0])/ dx)

def linear_get_upper_index(
    x: NDArray[float64],
    ub: float,
    *,
    dx: float | None = None,
) -> int:
    """
    Returns the integer, 'i', s.t. 'x[i] <= ub'. 
    """
    if not dx:
        dx = x[1] - x[0]
    return floor((ub - x[0])/ dx)

def linear_get_slice(
    x: NDArray[float64],
    lb: float,
    ub: float,
    *,
    dx: float | None = None,
) -> slice:
    if not dx:
        dx = x[1] - x[0]

    lb_idx = linear_get_lower_index(x, lb, dx=dx)
    ub_idx = linear_get_upper_index(x, ub, dx=dx)
    return slice(lb_idx, ub_idx + 1, 1)

def linear_get_mask(
    x: NDArray[float64],
    lb: float,
    ub: float,
    *,
    dx: float | None = None,
) -> NDArray[bool_]:
    mask = zeros(x.size, dtype=bool_)
    mask[linear_get_slice(x, lb, ub, dx=dx)] = True
    return mask

def logarithmic_get_lower_index(
    x: NDArray[float64],
    lb: float,
    *,
    v_res: float | None = None,
) -> int:
    if not v_res:
        v_res = x[1] / x[0] - 1

    return ceil()