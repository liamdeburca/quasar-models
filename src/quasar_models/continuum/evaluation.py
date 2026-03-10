from numpy import ones_like, zeros_like, log, float64, bool_
from numpy.typing import NDArray

def evaluate(
    x: float | NDArray[float64],
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> float | NDArray[float64]:
    return flux * (x / x0)**(-alpha)

def evaluate_sparse(
    x: float | NDArray[float64],
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> tuple[NDArray[bool_], NDArray[float64]]:
    return (
        ones_like(x, dtype=bool), 
        evaluate(x, flux, alpha, x0=x0),
    )

def fit_deriv(
    x: NDArray[float64],
    flux: float,
    alpha: float,
    x0: float = 1.0,
    fixed: dict[str, bool] = None,
) -> list[NDArray[float64]]:    
    df_dflux  = zeros_like(x, dtype=float64)
    df_dalpha = zeros_like(x, dtype=float64)

    if not (fixed is None) and not all(fixed.values()):
        _f = (x / x0)**(-alpha)
        if not fixed['flux']: 
            df_dflux[:] = _f
        if not fixed['alpha']: 
            df_dalpha[:] = _f * flux * log(x / x0) 

    return [df_dflux, df_dalpha]

def inverse(
    y: float | NDArray[float64],
    flux: float,
    alpha: float,
    x0: float = 1.0,
) -> float | NDArray[float64]:
    return x0 * (y / flux)**(-1 / alpha)