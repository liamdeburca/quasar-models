__all__ = [
    "evaluate",
    "fit_deriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
]

from typing import Protocol
from numpy import ones_like, stack, float64, bool_, zeros, zeros_like

from quasar_typing.numpy import FloatVector, BoolVector, FloatMatrix
from quasar_core.modelling import powerlaw

class EvaluateFunc(Protocol):
    def __call__(
        self,
        x: FloatVector, 
        flux: float, 
        alpha: float, 
        *,
        x0: float = 1.0,
    ) -> FloatVector:
        ...

class FitDerivFunc(Protocol):
    def __call__(
        self,
        derivs: FloatMatrix,
        x: FloatVector,
        flux: float,
        alpha: float,
    ) -> None:
        ...

###

def _evaluate(
    x: FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
    y: FloatVector | None = None
) -> FloatVector:
    if y is None:
        y = zeros_like(x, dtype=float64)
    return powerlaw.evaluate(y, x, flux, alpha, x0)

def evaluate_sparse(
    x: FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> tuple[BoolVector, FloatVector]:
    return (
        ones_like(x, dtype=bool_), 
        evaluate(x, flux, alpha, x0=x0),
    )

###

def choose_evaluate_func() -> EvaluateFunc:
    return _evaluate

def evaluate(
    x: FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
    evaluate_func: EvaluateFunc | None = None,
) -> FloatVector:
    if evaluate_func is None:
        evaluate_func = choose_evaluate_func()
    return evaluate_func(x, flux, alpha, x0=x0)

def _does_nothing(*args) -> None:
    pass

def choose_fit_deriv_func(
    fixed: dict[str, bool] | None,
) -> FitDerivFunc:
    if fixed is None:
        fixed = {'flux': False, 'alpha': False}

    match tuple(fixed.values()):
        case (False, False):
            return powerlaw.fit_deriv_all
        case (False, True):
            return powerlaw.fit_deriv_only_flux
        case (True, False):
            return powerlaw.fit_deriv_only_alpha
        case (True, True):
            return _does_nothing

def fit_deriv(
    x: FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
    fixed: dict[str, bool] | None = None,
    derivs: list[FloatVector] | FloatMatrix | None = None,
    fit_deriv_func: FitDerivFunc | None = None,
) -> list[FloatVector]:
    if derivs is None:
        derivs = zeros((2, x.size), dtype=float64)
    elif isinstance(derivs, list):
        derivs = stack(derivs, axis=0)

    if fit_deriv_func is None:
        fit_deriv_func = choose_fit_deriv_func(fixed)
    fit_deriv_func(derivs, x, flux, alpha, x0)

    return list(derivs)

def inverse(
    y: float | FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> float | FloatVector:
    return x0 * (y / flux)**(-1 / alpha)