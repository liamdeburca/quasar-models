"""
    Lorem ipsum.
"""
__all__ = ['evaluate', 'evaluate_sparse', 'fit_deriv']

from math import hypot
from numpy import exp, pi, float64, zeros

from quasar_typing.numpy import FloatVector, BoolVector, FloatMatrix

N_SIGMAS:  float = 3.0
GAUSS_AMP: float = 1 / (2 * pi)**0.5

def evaluate(
    x: float | FloatVector,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    gauss_amp: float = GAUSS_AMP,
) -> float | FloatVector:
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
    sigma = mean * hypot(sigma_v, sigma_res)
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

def evaluate_sparse(
    x: FloatVector,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    n_sigmas: float = N_SIGMAS,
    gauss_amp: float = GAUSS_AMP,
) -> tuple[BoolVector, FloatVector]:
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
    mean = wave * (1 + v_off)
    sigma = mean * hypot(sigma_v, sigma_res)

    mask = (mean - n_sigmas * sigma <= x) & (x <= mean + n_sigmas * sigma)

    inv_sigma = 1.0 / sigma
    z = (x[mask] - mean) * inv_sigma
    y = gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

    return mask, y

def fit_deriv_numba(
    x: FloatVector,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    gauss_amp: float = GAUSS_AMP,   
    fixed_strength: bool = True,
    fixed_sigma_v: bool = True,
    fixed_v_off: bool = True,
) -> list[FloatVector]:
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
    gauss_amp : float, optional

    fixed_strength : bool, optional
    fixed_sigma_v : bool, optional
    fixed_v_off : bool, optional

    Returns
    -------
    list[1D numpy.array of floats]

    Notes
    -----
    Lorem ipsum.
    """
    df_dstrength = zeros(x.size, dtype=float64)
    df_dsigma_v = zeros(x.size, dtype=float64)
    df_dv_off = zeros(x.size, dtype=float64)

    if not (fixed_strength and fixed_sigma_v and fixed_v_off):
        mean = wave * (1 + v_off)
        sigma_tot_sq = sigma_v * sigma_v + sigma_res * sigma_res
        sigma = mean * sigma_tot_sq**0.5
        inv_sigma = 1.0 / sigma
        z = (x - mean) * inv_sigma
        z_sq = z * z
        amp = gauss_amp * inv_sigma
        _f = amp * exp(-0.5 * z_sq)
        f = strength * _f

        if not fixed_strength:
            df_dstrength[:] = _f

        if not fixed_sigma_v:
            df_dsigma_v[:] = f * (z_sq - 1) * sigma_v / sigma_tot_sq

        if not fixed_v_off:
            df_dv_off[:] = f * (z * x * inv_sigma - 1) / (1 + v_off)

    return [df_dstrength, df_dsigma_v, df_dv_off]

def fit_deriv(
    x: FloatVector,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    fixed: dict[str, bool] | None = None,
    gauss_amp: float = GAUSS_AMP,   
) -> list[FloatVector]:
    """
    ** NUMBA OPTIMISED FUNCTION (FASTMATH) **

    Convenience function wrapping fit_deriv_numba.

    Parameters
    ----------
    x : 1D numpy.array of floats
    strength : float
    sigma_v : float
    v_off : float
    wave : float
    sigma_res : float
    fixed : dict[str, bool], optional
    gauss_amp : float, optional

    Returns
    -------
    list[1D numpy.array of floats]

    Notes
    -----
    Lorem ipsum.
    """
    if fixed is None:
        fixed = {'strength': False, 'sigma_v': False, 'v_off': False}

    return fit_deriv_numba(
        x,
        strength,
        sigma_v,
        v_off,
        wave,
        sigma_res,
        gauss_amp,
        fixed_strength=fixed['strength'],
        fixed_sigma_v=fixed['sigma_v'],
        fixed_v_off=fixed['v_off'],
    )

### Derivative w.r.t. x --  useful for numerical optimisation

def prime(
    x: float | FloatVector,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    gauss_amp: float = GAUSS_AMP,
) -> float | FloatVector:
    """
    ** NUMBA OPTIMISED FUNCTION (FASTMATH) **

    Derivative of evaluate() w.r.t. x.

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
    sigma = mean * hypot(sigma_v, sigma_res)
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return -z * inv_sigma * evaluate(
        x, 
        strength, sigma_v, v_off, 
        wave=wave, 
        sigma_res=sigma_res, 
        gauss_amp=gauss_amp,
    )