"""
File containing utilities for stop conditions.
"""
from numpy import diff, finfo, ndarray
from quasar_typing.numpy import FloatMatrix, FloatVector
from scipy.linalg import norm

PRECISION: float = finfo(float).eps


def fcrit_stop(
    fcrit: float,
    f: float,
) -> bool:
    """
    Check whether the cost function value is below a critical threshold.
    """
    return abs(f) < abs(fcrit)


def ftol_stop(
    ftol: float,
    df: float,
    f: float,
) -> bool:
    """
    Check whether the relative decrease in the cost function is below the 
    tolerance, 'ftol'.

    For further details, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    Parameters
    ----------
    ftol : float
        The relative tolerance for the decrease in the cost function.
    df : float
        The change in the cost function value between iterations.
    f : float
        The current value of the cost function.

    Returns
    -------
    bool
        True if the relative decrease in the cost function is below 'ftol',
        indicating that the optimization should stop; False otherwise.

    Notes
    -----
    The used 'ftol' parameter is the maximum of the input 'ftol' and the machine 
    precision for double precision floating point numbers. 
    """
    ftol = max(ftol, PRECISION)
    return abs(df) < ftol * abs(f)


def xtol_stop(
    xtol: float,
    dx: float | FloatVector,
    x: float | FloatVector,
) -> bool:
    """
    Check whether the relative change in the parameters is below the tolerance,
    'xtol'.

    For further details, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    Parameters
    ----------
    xtol : float
        The relative tolerance for the change in the parameters.
    dx : float or FloatVector
        The change in the parameters between iterations.
    x : float or FloatVector
        The current value of the parameters.

    Returns
    -------
    bool
        True if the relative change in the parameters is below 'xtol',
        indicating that the optimization should stop; False otherwise.

    Notes
    -----
    The used 'xtol' parameter is the maximum of the input 'xtol' and the machine 
    precision for double precision floating point numbers. 
    """
    xtol = max(xtol, PRECISION)
    if isinstance(dx, float) and isinstance(x, float):
        return abs(dx) < xtol * abs(x)
    elif isinstance(dx, ndarray) and isinstance(x, ndarray):
        return norm(dx) < xtol * norm(x)
    raise ValueError("`dx` and `x` must be both floats or both numpy arrays.")


def fcrit_stop_batched(
    fcrit: float,
    f: FloatVector,
) -> bool:
    """
    Check whether the 'fcrit' stop condition is satisfied anywhere in an array of 
    cost function values.
    """
    return any(fcrit_stop(fcrit, _f) for _f in f)


def ftol_stop_batched(
    ftol: float,
    f: FloatVector,
    *,
    df: FloatVector | None = None,
) -> bool:
    """
    Check whether the 'ftol' stop condition is satisfied anywhere in an array of 
    cost function values.
    """
    if df is None:
        df = diff(f)

    return any(ftol_stop(ftol, _df, _f) for _df, _f in zip(df, f[:-1]))


def xtol_stop_batched(
    xtol: float,
    x: FloatVector | FloatMatrix,
    dx: FloatVector | FloatMatrix | None = None,
) -> bool:
    """
    Check whether the 'xtol' stop condition is satisfied anywhere in an array of
    parameter values.

    The shape of 'x' should be (n_samples,) or (n_samples, n_parameters).
    """
    if dx is None:
        dx = diff(x, axis=0)

    return any(xtol_stop(xtol, _dx, _x) for _dx, _x in zip(dx, x[:-1]))