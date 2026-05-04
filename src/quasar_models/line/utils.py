"""
    Lorem ipsum.
"""

from math import pi
from numpy import argwhere, argmax, argmin, invert, dot, float64
from numpy.typing import NDArray
from quasar_typing.bounds import AstropyBounds

from ..utils.astropy import apply_bounds

def measure_intersection(
    x: NDArray[float64],
    y: NDArray[float64],
    y_crit: float,
) -> float:
    """
    Lorem ipsum.

    Parameters
    ----------
    x : 1D numpy.array of floats
    y : 1D numpy.array of floats
    y_crit : float

    Returns
    -------
    float

    Notes
    -----
    Lorem ipsum.
    """
    match x.size:
        case 0:
            # No datapoints
            raise ValueError("Input arrays must not be empty.")
        case 1:
            # Only a single datapoint
            return x[0]
        case 2:
            # Only two datapoints
            x1, x2 = x
            y1, y2 = y
        case _:
            mask = (y[:-1] >= y_crit) & (y[1:] < y_crit)
            if not mask.any(): return x[-1]

            n = argwhere(mask).min()

            # print(x.size, x[0], x[-1], n, x[n:n+2])
            x1, x2 = x[n:n+2]
            y1, y2 = y[n:n+2]

    return (x2 - x1) / (y2 - y1) * (y_crit - y1) + x1

def measure_sigma(
    wave: float,
    x: NDArray[float64],
    y: NDArray[float64],
    r: float = 0.5,
) -> float:
    """
    Lorem ipsum.

    Parameters
    ----------
    wave : float
    x : 1D numpy.array of floats
    y : 1D numpy.array of floats
    r : float, optional

    Returns
    -------
    float

    Notes
    -----
    Lorem ipsum.
    """
    y_crit: float = r * y[argmin(abs(x - wave))]

    is_left = (x < wave)
    if is_left.any():
        x_left  = x[is_left][::-1]
        y_left  = y[is_left][::-1]
        x1 = measure_intersection(x_left, y_left, y_crit)
    else:
        x1 = x[0]

    is_right = invert(is_left)
    if is_right.any():
        x_right  = x[is_right]
        y_right  = y[is_right]
        x2 = measure_intersection(x_right, y_right, y_crit)
    else:
        x2 = x[-1]

    return (x2 - x1) / wave / (2 * (-2 * pi * r)**0.5)

def instantiate_model(
    line: float, 
    x: NDArray[float64], 
    y: NDArray[float64], 
    y_smooth: NDArray[float64] | None = None,
    sigma_res: float = 0,
    v_off_bounds:    AstropyBounds = (-1, 1),
    sigma_v_bounds:  AstropyBounds = (0, None),
    strength_bounds: AstropyBounds = (0, None)
) -> tuple[float, float, float]:
    """
    Lorem ipsum.

    Parameters
    ----------
    line : float
    x : 1D numpy.array of floats
    y : 1D numpy.array of floats
    y_smooth : 1D numpy.array of floats or None, optional
    sigma_res : float, optional
    v_off_bounds : tuple of floats, optional
    sigma_v_bounds : tuple of floats, optional
    strength_bounds : tuple of floats, optional

    Returns
    -------
    tuple[float, float, float]

    Notes
    -----
    Lorem ipsum.
    """
    if y_smooth is None: y_smooth = y

    # Guessing the velocity offset 
    # - Method 4 (lmfit) from thesis (smoothed preferred)
    mask = y_smooth > 0.5 * (y_smooth.min() + y_smooth.max())
    
    _x = x[mask]
    if _x.size <= 1: _mu = x[argmax(y_smooth)]
    else:            _mu = _x.mean()

    v_off = apply_bounds((_mu - line) / line, v_off_bounds)
    mu = line * (1 + v_off)

    # Guessing the velocity dispersion
    # - Method 4 (FWQM) from thesis (smoothed preferred)
    sigma = measure_sigma(mu, x, y_smooth, 0.25)
    sigma_v = max(sigma**2 - sigma_res**2, 0)**0.5

    # Guessing the line strength
    # - Method 2 (integral) from thesis (raw preferred)
    strength = dot(y, x * sigma_res)

    return (
        apply_bounds(strength, strength_bounds),
        apply_bounds(sigma_v, sigma_v_bounds),
        v_off,
    )