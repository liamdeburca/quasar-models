"""
    Lorem ipsum.
"""
__all__ = ['evaluate', 'evaluate_sparse', 'fit_deriv']

from numpy import exp, pi, hypot, zeros_like, float64, bool_
from numpy.typing import NDArray

N_SIGMAS:  float = 3.0
GAUSS_AMP: float = 1 / (2 * pi)**0.5

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
    sigma = mean * hypot(sigma_v, sigma_res)
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

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
    mean = wave * (1 + v_off)
    sigma = mean * hypot(sigma_v, sigma_res)

    mask = (mean - n_sigmas * sigma <= x) & (x <= mean + n_sigmas * sigma)

    inv_sigma = 1.0 / sigma
    z = (x[mask] - mean) * inv_sigma
    y = gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

    return mask, y

def fit_deriv_numba(
    x: NDArray[float64],
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    gauss_amp: float = GAUSS_AMP,   
    fixed_strength: bool = True,
    fixed_sigma_v: bool = True,
    fixed_v_off: bool = True,
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
    df_dstrength = zeros_like(x, dtype=float64)
    df_dsigma_v  = zeros_like(x, dtype=float64)
    df_dv_off    = zeros_like(x, dtype=float64)

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
    x: NDArray[float64],
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    fixed: dict[str, bool] | None = None,
    gauss_amp: float = GAUSS_AMP,   
) -> list[NDArray[float64]]:
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
        fixed = {'strength': True, 'sigma_v': True, 'v_off': True}

    return fit_deriv_numba(
        x,
        strength,
        sigma_v,
        v_off,
        wave,
        sigma_res,
        gauss_amp,
        fixed_strength=fixed.get('strength', True),
        fixed_sigma_v=fixed.get('sigma_v', True),
        fixed_v_off=fixed.get('v_off', True),
    )

### Derivative w.r.t. x --  useful for numerical optimisation

def prime(
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