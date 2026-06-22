from math import log as math_log
from numpy import ones_like, exp, log, float64, bool_, zeros

from quasar_typing.numpy import FloatVector, BoolVector

def evaluate(
    x: float | FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> float | FloatVector:
    return flux * exp(alpha * (log(x) - math_log(x0)))

def evaluate_sparse(
    x: float | FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> tuple[BoolVector, FloatVector]:
    return (
        ones_like(x, dtype=bool_), 
        evaluate(x, flux, alpha, x0=x0),
    )

def fit_deriv(
    x: FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    if fixed is None:
        fixed = {'flux': False, 'alpha': False}

    df_dflux = zeros(x.size, dtype=float64)
    df_dalpha = zeros(x.size, dtype=float64)

    if not all(fixed.values()):
        _f = (x / x0)**alpha
        if not fixed['flux']: 
            df_dflux[:] = _f
        if not fixed['alpha']: 
            df_dalpha[:] = flux * _f * log(x / x0)

    return [df_dflux, df_dalpha]

def inverse(
    y: float | FloatVector,
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> float | FloatVector:
    return x0 * (y / flux)**(-1 / alpha)