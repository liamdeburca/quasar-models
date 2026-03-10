from numpy import zeros_like, expm1, exp, log, float64
from numpy.typing import NDArray

def evaluate(
    x: float | NDArray[float64], 
    tau: float, 
    scale: float, 
    *,
    edge: float = None,
) -> float | NDArray[float64]:
    """
    Exponential attenuation at the Balmer edge.
    """
    if tau == 0: return zeros_like(x, dtype=float64)
    else:        return expm1(-tau * (x / edge)**scale) / expm1(-tau)

def fit_deriv(
    x: NDArray[float64], 
    tau: float, 
    scale: float, 
    *,
    edge: float = None,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    """
    Normalised exponential attenuation's derivatives.
    """
    df_dtau = zeros_like(x, dtype=float64)
    df_dscale = zeros_like(x, dtype=float64)

    if not (fixed is None) and not all(fixed.values()) and (tau != 0):
        r = (x / edge)**scale
        s = tau * r
        exp_s = exp(-s)
        exp_t = exp(-tau)
        denom = 1 - exp_t

        if not fixed['tau']:
            f = (1 - exp_s) / denom
            df_dtau[:] = (r * exp_s - f * exp_t) / denom

        if not fixed['scale']:
            df_dscale[:] = s * exp_s / denom * log(x / edge)

    return [df_dtau, df_dscale]