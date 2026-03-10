"""
    Lorem ipsum.
"""

from numpy import exp, pi, zeros_like, float64, bool_
from numpy.typing import NDArray
from numba import njit

N_SIGMAS:  float = 3.0
GAUSS_AMP: float = 1 / (2 * pi)**0.5

@njit(fastmath=True)
def evaluate(
    x: float | NDArray[float64],
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    gauss_amp: float = GAUSS_AMP,
) -> float | NDArray[float64]:
    """
    ** NUMBA OPTIMISED FUNCTION (FASTMATH) **

    Lorem ipsum.

    Parameters
    ----------
    x : float or 1D numpy.array of floats
    strength : float
    sigma_v : float
    v_off : float
    wave : float, optional
    sigma_res : float, optional
    gauss_amp : float, optional

    Returns
    -------
    float or 1D numpy.array of floats

    Notes
    -----
    Lorem ipsum.
    """
    mean = wave * (1 + v_off)
    sigma = mean * (sigma_v**2 + sigma_res**2)**0.5
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

@njit(fastmath=True)
def evaluate_sparse(
    x: NDArray[float64],
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    n_sigmas: float = N_SIGMAS,
    gauss_amp: float = GAUSS_AMP,
) -> tuple[NDArray[bool_], NDArray[float64]]:
    """
    ** NUMBA OPTIMISED FUNCTION (FASTMATH) **

    Lorem ipsum.

    Parameters
    ----------
    x : 1D numpy.array of floats
    strength : float
    sigma_v : float
    v_off : float
    wave : float, optional
    sigma_res : float, optional
    n_sigmas : float, optional
    gauss_amp : float, optional

    Returns
    -------
    tuple[1D numpy.array of bools, 1D numpy.array of floats]

    Notes
    -----
    Lorem ipsum.
    """
    mean:  float = wave * (1 + v_off)
    sigma: float = mean * (sigma_v**2 + sigma_res**2)**0.5

    mask = (mean - n_sigmas * sigma <= x) & (x <= mean + n_sigmas * sigma)

    inv_sigma = 1.0 / sigma
    z = (x[mask] - mean) * inv_sigma

    return mask, gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

@njit(fastmath=True)
def fit_deriv(
    x: NDArray[float64],
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    fixed: dict[str, bool] | None,
    gauss_amp: float = GAUSS_AMP,   
) -> list[NDArray[float64]]:
    """
    ** NUMBA OPTIMISED FUNCTION (FASTMATH) **

    Lorem ipsum.

    Parameters
    ----------
    x : 1D numpy.array of floats
    strength : float
    sigma_v : float
    v_off : float
    wave : float
    sigma_res : float
    fixed : dict of str to bool or None
    gauss_amp : float, optional

    Returns
    -------
    list[1D numpy.array of floats]

    Notes
    -----
    Lorem ipsum.
    """
    df_dstrength = zeros_like(x, dtype=float64)
    df_dsigma_v  = zeros_like(x, dtype=float64)
    df_dv_off    = zeros_like(x, dtype=float64)

    if not (fixed is None) and not all(fixed.values()):
        mean = wave * (1 + v_off)
        sigma = mean * (sigma_v**2 + sigma_res**2)**0.5
        inv_sigma = 1.0 / sigma
        z = (x - mean) * inv_sigma
        z_sq = z**2
        _f = gauss_amp * exp(-0.5 * z * z) * inv_sigma
        f = strength * _f

        if not fixed['strength']:
            df_dstrength[:] = _f

        if not fixed['sigma_v']:
            df_dsigma_v[:] = f * (z_sq - 1) * sigma_v * (mean * inv_sigma)**2

        if not fixed['v_off']:
            df_dv_off[:] = f * (z * x * inv_sigma - 1) / (1 + v_off)

    return [df_dstrength, df_dsigma_v, df_dv_off]