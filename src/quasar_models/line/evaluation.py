"""
    Lorem ipsum.
"""
__all__ = [
    "evaluate",
    "fit_deriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
]

from typing import Protocol
from math import hypot, pi, log, sqrt
from numpy import float64, zeros, zeros_like, stack

from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_core.modelling import gaussian

N_SIGMAS:       float = 3.0
GAUSS_AMP:      float = 1 / (2 * pi)**0.5
SIGMA_TO_FWHM:  float = 2 * sqrt(2 * log(2))  # ≈ 2.3548
FWHM_TO_SIGMA:  float = 1 / SIGMA_TO_FWHM     # ≈ 0.4247

class EvaluateFunc(Protocol):
    def __call__(
        self,
        x: FloatVector,
        strength: float,
        fwhm_v: float,
        v_off: float,
        wave: float = 1.0,
        sigma_res: float = 0.0,
        y: FloatVector | None = None,
    ) -> FloatVector:
        ...

class FitDerivFunc(Protocol):
    def __call__(
        self,
        derivs: FloatMatrix,
        x: FloatVector,
        strength: float,
        fwhm_v: float,
        v_off: float,
        wave: float,
        sigma_res: float,
    ) -> None:
        ...

###

def _evaluate(
    x: FloatVector,
    strength: float,
    fwhm_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 0.0,
    y: FloatVector | None = None,
) -> float | FloatVector:
    """
    Lorem ipsum.

    Parameters
    ----------
    x : float or 1D numpy.array of floats
    strength : float
    fwhm_v : float
    v_off : float
    wave : float, optional
    sigma_res : float, optional

    Returns
    -------
    float or 1D numpy.array of floats

    Notes
    -----
    Lorem ipsum.
    """
    if y is None:
        y = zeros_like(x, dtype=float64)
    return gaussian.evaluate_v(y, x, strength, fwhm_v, v_off, wave, sigma_res)

###

def choose_evaluate_func() -> EvaluateFunc:
    return _evaluate

def evaluate(
    x: FloatVector,
    strength: float,
    fwhm_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 0.0,
    y: FloatVector | None = None,
    evaluate_func: EvaluateFunc | None = None,
) -> FloatVector:
    if evaluate_func is None:
        evaluate_func = choose_evaluate_func()
    return evaluate_func(
        x, 
        strength, fwhm_v, v_off, 
        wave=wave, sigma_res=sigma_res,
        y=y,
    )

def _does_nothing(*args):
    pass

def choose_fit_deriv_func(
    fixed: dict[str, bool] | None,
) -> FitDerivFunc:
    if fixed is None:
        fixed = {'strength': False, 'fwhm_v': False, 'v_off': False}


    match tuple(fixed.values()):
        case (False, False, False):
            return gaussian.fit_deriv_v_all
        case (False, False, True):
            return gaussian.fit_deriv_v_strength_and_fwhm_v
        case (False, True, False):
            return gaussian.fit_deriv_v_strength_and_v_off
        case (True, False, False):
            return gaussian.fit_deriv_v_fwhm_v_and_v_off
        case (False, True, True):
            return gaussian.fit_deriv_v_only_strength
        case (True, False, True):
            return gaussian.fit_deriv_v_only_fwhm_v
        case (True, True, False):
            return gaussian.fit_deriv_v_only_v_off
        case (True, True, True):
            return _does_nothing

def fit_deriv(
    x: FloatVector,
    strength: float,
    fwhm_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    fixed: dict[str, bool] | None = None,
    derivs: list[FloatVector] | FloatMatrix | None = None,
    fit_deriv_func: FitDerivFunc | None = None,
) -> list[FloatVector]:
    """
    Convenience function wrapping fit_deriv_numba.

    Parameters
    ----------
    x : 1D numpy.array of floats
    strength : float
    fwhm_v : float
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
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    elif isinstance(derivs, list):
        derivs = stack(derivs, axis=0)

    if fit_deriv_func is None:
        fit_deriv_func = choose_fit_deriv_func(fixed)

    fit_deriv_func(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

    return list(derivs)

### Derivative w.r.t. x --  useful for numerical optimisation

def prime(
    x: float | FloatVector,
    strength: float,
    fwhm_v: float,
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
    fwhm_v : float
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
    sigma_v = fwhm_v * FWHM_TO_SIGMA
    mean = wave * (1 + v_off)
    sigma = mean * hypot(sigma_v, sigma_res)
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return -z * inv_sigma * evaluate(
        x, 
        strength, fwhm_v, v_off, 
        wave=wave, 
        sigma_res=sigma_res, 
        gauss_amp=gauss_amp,
    )