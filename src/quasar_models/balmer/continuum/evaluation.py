__all__ = ['evaluate', 'fit_deriv']

from numpy import zeros_like, expm1, float64, exp, log
from quasar_typing.numpy import FloatVector
from quasar_utils.convolution import convolve_signal, kernel, kernel_deriv

# ATTENUATION

def attenuation_evaluate(
    x: float | FloatVector, 
    tau: float, 
    scale: float, 
    *,
    edge: float,
    out: FloatVector | None = None,
) -> float | FloatVector:
    """
    Exponential attenuation at the Balmer edge.
    """
    if out is None:
        out = zeros_like(x, dtype=float64)

    if tau == 0:
        return out
    
    mask = (x <= edge)
    out[mask] = -expm1(-tau * (x[mask] / edge)**scale)
    
    return out

def attenuation_fit_deriv(
    x: FloatVector, 
    tau: float, 
    scale: float, 
    *,
    edge: float,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    """
    Exponential attenuation's derivatives.
    """
    df_dtau = zeros_like(x, dtype=float64)
    df_dscale = zeros_like(x, dtype=float64)

    mask = (x <= edge)
    if mask.any() and (fixed is not None) and not all(fixed.values()):
        _x = x[mask]
        r = (_x / edge)**scale
        s = tau * r
        exp_s = exp(-s)

        if not fixed['tau']:
            df_dtau[mask] = -r * exp_s

        if not fixed['scale'] and tau != 0:
            df_dscale[mask] = -s * exp_s * log(_x / edge)

    return [df_dtau, df_dscale]

# CONTINUUM

def continuum_evaluate(
    x: float | FloatVector,
    flux: float,
    temp: float,
    *,
    edge: float,
    boltz: float,
    out: FloatVector | None = None,
) -> float | FloatVector:
    """
    The continuum (Planck function) contribution to the total Balmer continuum.
    """ 
    if out is None:
        out = zeros_like(x, dtype=float64)

    if flux == 0:
        return out
    
    mask = (x <= edge)
    out[mask] = flux / (x[mask]**5 * expm1(boltz / (temp * x[mask])))
    return out
    
def continuum_fit_deriv(
    x: FloatVector,
    flux: float,
    temp: float,
    *,
    edge: float | None = None,
    boltz: float | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    """
    The continuum (Planck function) contribution's derivative(s).
    """    
    df_dflux = zeros_like(x, dtype=float64)
    df_dtemp = zeros_like(x, dtype=float64)

    mask = (x <= edge)

    if mask.any() and (fixed is not None) and not all(fixed.values()):
        _x = x[mask]
        y = boltz / (temp * _x)
        exp_y = exp(y)
        expm1_y = expm1(y)

        if not fixed['flux']:
            df_dflux[mask] = 1 / (_x**5 * expm1_y)

        if not fixed['temp'] and flux != 0:
            df_dtemp[mask] = flux * exp_y * boltz / (temp**2 * _x**6 * expm1_y**2)
    
    return [df_dflux, df_dtemp]

# CONTINUUM + ATTENUATION + BROADENING

def evaluate(
    x: FloatVector,
    flux: float,
    fwhm: float,
    temp: float,
    tau: float,
    scale: float,
    *,
    sigma_res: float,
    edge: float,
    boltz: float,
    normalisation: float | None = None,
) -> FloatVector:
    f = continuum_evaluate(x, flux, temp, edge=edge, boltz=boltz)
    f *= attenuation_evaluate(x, tau, scale, edge=edge)
    f = convolve_signal.__wrapped__(f, kernel(fwhm, sigma_res))

    return f if normalisation is None else f / normalisation

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    temp: float,
    tau: float,
    scale: float,
    *,
    sigma_res: float,
    edge: float,
    boltz: float,
    normalisation: float | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    
    df_dflux = zeros_like(x, dtype=float)
    df_dfwhm = zeros_like(x, dtype=float)
    df_dtemp = zeros_like(x, dtype=float)
    df_dtau = zeros_like(x, dtype=float)
    df_dscale = zeros_like(x, dtype=float)

    if (fixed is None) or all(fixed.values()):
        return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale]
    
    normalisation = normalisation or 1.0
        
    _k = None
    def k() -> FloatVector:
        nonlocal _k
        if _k is None: 
            _k = kernel.__wrapped__(fwhm, sigma_res)
        return _k
    
    _dk = None
    def dk() -> FloatVector: 
        nonlocal _dk
        if _dk is None: 
            _dk = kernel_deriv.__wrapped__(fwhm, sigma_res)
        return _dk

    _cont = None
    def cont() -> FloatVector:
        nonlocal _cont
        if _cont is None:
            _cont = continuum_evaluate(
                x, flux, temp,
                boltz=boltz,
            )
        return _cont

    _dcont = None
    def dcont() -> list[FloatVector]:
        nonlocal _dcont
        if _dcont is None:
            _dcont = continuum_fit_deriv(
                x, flux, temp,
                boltz=boltz,
                fixed={p: fixed[p] for p in ['flux', 'temp']}
            )
        return _dcont

    _atte = None
    def atte() -> FloatVector:
        nonlocal _atte
        if _atte is None:
            _atte = attenuation_evaluate(
                x, tau, scale,
                edge=edge,
            )
        return _atte    

    _datte = None
    def datte() -> list[FloatVector]:
        nonlocal _datte
        if _datte is None:
            _datte = attenuation_fit_deriv(
                x, tau, scale,
                edge=edge,
                fixed={p: fixed[p] for p in ['tau', 'scale']},
            )
        return _datte  

    if not fixed['flux']:
        df_dflux[:] = convolve_signal.__wrapped__(dcont()[0] * atte(), k())
        df_dflux[:] /= normalisation

    if not fixed['fwhm'] and flux != 0:
        df_dfwhm[:] = convolve_signal.__wrapped__(cont() * atte(), dk())
        df_dfwhm[:] /= normalisation

    if not fixed['temp'] and flux != 0:
        df_dtemp[:] = convolve_signal.__wrapped__(dcont()[1] * atte(), k())
        df_dtemp[:] /= normalisation

    if not fixed['tau'] and flux != 0:
        df_dtau[:] = convolve_signal.__wrapped__(cont() * datte()[0], k())
        df_dtau[:] /= normalisation

    if not fixed['scale'] and flux != 0:
        df_dscale[:] = convolve_signal.__wrapped__(cont() * datte()[1], k())
        df_dscale[:] /= normalisation

    return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale]