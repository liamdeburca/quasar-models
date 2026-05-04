from numpy import pi, log, exp, zeros_like, float64
from numpy.typing import NDArray

GAUSS_AMP: float = 1 / (2 * pi)**0.5
FWHM_TO_SIGMA: float = 1 / (2 * (2 * log(2))**0.5)

def evaluate(
    x: float | NDArray[float64],
    lumin: float,
    fwhm: float,
    *,
    sigma_res: float = None, 
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
    gauss_amp: float = GAUSS_AMP,
) -> float | NDArray[float64]:
    """
    Summed velocity-space Gaussian contributions, vectorised over all lines.
    """
    if lumin == 0: return zeros_like(x, dtype=float64)

    denom = log(1 + sigma_res)
    _sigma = fwhm_to_sigma * fwhm / sigma_res
    inv_sigma = 1.0 / _sigma

    _x = log(x) / denom                        # (M,)
    _mu = log(waves) / denom                    # (N,)
    z = (_x - _mu[:, None]) * inv_sigma         # (N, M)

    signal = gauss_amp * inv_sigma \
        * (weights[:,None] * exp(-0.5 * z * z)).sum(axis=0)

    return lumin * signal

def fit_deriv(
    x: NDArray[float64],
    lumin: float,
    fwhm: float,
    *,
    sigma_res: float = None, 
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    gauss_amp: float = GAUSS_AMP,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    """
    Vectorised partial derivatives of the summed Balmer series Gaussians.
    """
    df_dlumin = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)

    if not (fixed is None) and not all(fixed.values()):
        denom = log(1 + sigma_res)
        _sigma = fwhm_to_sigma * fwhm / sigma_res
        inv_sigma = 1.0 / _sigma

        _x = log(x) / denom                        # (M,)
        _mu = log(waves) / denom                    # (N,)
        z = (_x - _mu[:, None]) * inv_sigma         # (N, M)
        z_sq = z * z                                # (N, M)

        gauss_vals = gauss_amp * inv_sigma \
            * weights[:, None] * exp(-0.5 * z_sq)  # (N, M)
        signal = gauss_vals.sum(axis=0)             # (M,)

        if not fixed['lumin']:
            df_dlumin[:] = 1 if (lumin == 0) else signal

        if not fixed['fwhm']:
            df_dfwhm[:] = lumin * fwhm_to_sigma / (sigma_res * _sigma) \
                * (gauss_vals * (z_sq - 1)).sum(axis=0)

    return [df_dlumin, df_dfwhm]