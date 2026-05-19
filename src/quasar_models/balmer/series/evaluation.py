from numpy import pi, log, exp, zeros_like, float64
from numpy.typing import NDArray
from quasar_typing.numpy import FloatVector

GAUSS_AMP: float = 1 / (2 * pi)**0.5
FWHM_TO_SIGMA: float = 1 / (2 * (2 * log(2))**0.5)
    
def evaluate(
    x: float | FloatVector,
    flux: float,
    fwhm: float,
    *,
    sigma_res: float = None, 
    edge: float = None,
    waves: FloatVector = None,
    weights: FloatVector = None,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
    gauss_amp: float = GAUSS_AMP,
    
    log_x: FloatVector | None = None,
    log_edge: FloatVector | None = None,
    log_waves: FloatVector | None = None,
    
    normalisation: float | None = 1.0,
) -> float | FloatVector:
    """
    Summed velocity-space Gaussian contributions normalised to 'flux' at the 
    Balmer edge.
    """
    assert fwhm > 0
    
    if flux == 0: 
        return zeros_like(x, dtype=float64)
        
    if log_x is None:
        log_x = log(x)
    if log_waves is None:
        log_waves = log(waves)

    denom = log(1 + sigma_res)
    _sigma = fwhm_to_sigma * fwhm / sigma_res
    inv_sigma = 1.0 / _sigma
    amp = gauss_amp * inv_sigma

    _x = log_x / denom
    _mu = log_waves / denom
    z = (_x - _mu[:, None]) * inv_sigma

    signal = amp * (weights[:,None] * exp(-0.5 * z * z)).sum(axis=0)

    if normalisation is None:
        if log_edge is None:
            log_edge = log(edge)
        _edge = log_edge / denom
        z_edge = (_edge - _mu) * inv_sigma
        normalisation = amp * (weights * exp(-0.5 * z_edge * z_edge)).sum()

    return flux * signal / normalisation

def fit_deriv(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    sigma_res: float = None, 
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    edge: float = None,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
    gauss_amp: float = GAUSS_AMP,

    log_x: FloatVector | None = None,
    log_edge: FloatVector | None = None,
    log_waves: FloatVector | None = None,

    normalisation: float | None = 1.0,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    """
    Vectorised partial derivatives of the summed Balmer series Gaussians.
    Derivatives with respect to normalized signal: signal(edge) = 1.
    """
    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)

    if (fixed is not None) and not all(fixed.values()):
        if log_x is None:
            log_x = log(x)
        if log_waves is None:
            log_waves = log(waves)
        if log_edge is None:
            log_edge = log(edge)

        denom = log(1 + sigma_res)
        _sigma = fwhm_to_sigma * fwhm / sigma_res
        inv_sigma = 1.0 / _sigma
        amp = gauss_amp * inv_sigma

        _x = log_x / denom
        _mu = log_waves / denom
        z = (_x - _mu[:, None]) * inv_sigma
        z_sq = z * z

        _edge = log_edge / denom
        z_edge = (_edge - _mu) * inv_sigma
        z_edge_sq = z_edge * z_edge

        exp_z_sq = exp(-0.5 * z_sq)
        exp_z_edge_sq = exp(-0.5 * z_edge_sq)
        
        signal = amp * (weights[:,None] * exp_z_sq).sum(axis=0)

        if normalisation is None:
            normalisation = amp * (weights * exp_z_edge_sq).sum()

        if not fixed['flux']:
            df_dflux[:] = signal / normalisation

        if not fixed['fwhm'] and flux != 0:
            dsignal_dfwhm = (fwhm_to_sigma / (sigma_res * _sigma)) \
                * (amp * (weights[:,None] * z_sq * exp_z_sq).sum(axis=0) - signal)
            
            dnorm_dfwhm = (fwhm_to_sigma / (sigma_res * _sigma)) \
                * (amp * (weights * z_edge_sq * exp_z_edge_sq).sum() - normalisation)
            
            df_dfwhm[:] = flux * \
                (normalisation * dsignal_dfwhm - signal * dnorm_dfwhm) \
                    / normalisation**2

    return [df_dflux, df_dfwhm]