from numpy import pi, log, exp, zeros_like, float64
from numpy.typing import NDArray

GAUSS_AMP: float = 1 / (2 * pi)**0.5
FWHM_TO_SIGMA: float = 1 / 2 * (2 * log(2))**0.5

def _g(
    x: float | NDArray[float64],
    A: float,
    mu: float,
    sigma: float,
    gauss_amp: float = GAUSS_AMP,
) -> float | NDArray[float64]:
    inv_sigma = 1 / sigma
    z = (x - mu) * inv_sigma
    return gauss_amp * A * exp(0.5 * z**2) * inv_sigma

def evaluate(
    x: float | NDArray[float64],
    lumin: float,
    fwhm: float,
    *,
    sigma_res: float = None, 
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
) -> float | NDArray[float64]:
    """
    Summed velocity-space Gaussian contributions.
    """
    if lumin == 0: return zeros_like(x, dtype=float64)

    denom = log(1 + sigma_res)
    _x = log(x) / denom
    _sigma = fwhm_to_sigma * fwhm / sigma_res
    signal = sum(
        _g(_x, *args, _sigma) \
        for args in zip(weights, log(waves) / denom)
    )

    return lumin * signal

def fit_deriv(
    x: NDArray[float64],
    lumin: float,
    fwhm: float,
    *,
    sigma_res: float = None, 
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    """
    Summed and normalised velocity-space Gaussian contributions' combined 
    derivative.
    """
    df_dlumin = zeros_like(x, dtype=float)
    df_dfwhm = zeros_like(x, dtype=float)

    if not (fixed is None) and not all(fixed.values()):
        denom: float = log(1 + sigma_res)

        _x: NDArray[float64] = log(x) / denom
        _sigma: float = fwhm_to_sigma * fwhm / sigma_res
        _mu: NDArray[float64] = log(waves) / denom

        signal_elems: list[NDArray[float64]] = [
            _g(_x, *args, _sigma) \
            for args in zip(weights, _mu)
        ]

        if not fixed['lumin']:
            df_dlumin[:] = sum(signal_elems)

        if not fixed['fwhm'] and (lumin != 0):
            df_dfwhm[:] = lumin * sum(
                g / _sigma * (((_x - m) / _sigma)**2 - 1) \
                    * fwhm_to_sigma / sigma_res \
                for (g, m) in zip(signal_elems, _mu)
            )

    return [df_dlumin, df_dfwhm]