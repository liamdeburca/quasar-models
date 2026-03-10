from numpy import zeros_like, expm1, float64, exp
from numpy.typing import NDArray

def evaluate(
    x: float | NDArray[float64],
    flux: float,
    temp: float,
    *,
    edge: float | None = None,
    boltz: float | None = None,
) -> float | NDArray[float64]:
    """
    The continuum (Planck function) contribution to the total Balmer continuum.
    """    
    if flux == 0: return zeros_like(x, dtype=float64)
    b: float = boltz / temp
    return flux * (edge / x)**5 * expm1(b / edge) / expm1(b / x)

def fit_deriv(
    x: NDArray[float64],
    flux: float,
    temp: float,
    *,
    edge: float | None = None,
    boltz: float | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    """
    The continuum (Planck function) contribution's derivative(s).
    """    
    df_dflux: NDArray[float64] = zeros_like(x, dtype=float)
    df_dtemp: NDArray[float64] = zeros_like(x, dtype=float)

    if not (fixed is None) and not all(fixed.values()):
        mask = (x <= edge)
        _x = x[mask]
        
        b = boltz / temp
        exp_y = exp(y := b / _x)
        exp_y_BE = exp(y_BE := b / edge)
        f_over_flux = (edge / _x)**5 * (exp_y - 1)**-1 * (exp_y_BE - 1)

        # Partial derivatives
        if not fixed['flux']:
            df_dflux[mask] = f_over_flux

        if not fixed['temp'] and (flux != 0):
            df_dtemp[mask] = flux * f_over_flux / temp * (
                y * exp_y / (exp_y - 1) \
                - y_BE * exp_y_BE / (exp_y_BE - 1)
            )
    
    return [df_dflux, df_dtemp]