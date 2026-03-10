from numpy import zeros_like, dot, interp, ndarray, float64
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from quasar_utils.convolution import (
    convolve_signal, kernel, kernel_deriv, _identify_closest_idx,
)
from .attenuation import evaluation as attenuation
from .continuum import evaluation as continuum
from .series import evaluation as series

from ..utils.template import Template

def evaluate(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    temp: float,
    tau: float,
    scale: float,
    ratio: float,
    *,
    sigma_res: float = None,
    edge: float = None,
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    boltz: float = None,
    x_grid: NDArray[float64] = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> float | NDArray[float64]:
    """
    Evaluates the entire Balmer pseudo-continuum model. First, the continuum
    (blackbody + attenuation) contribution is calculated over an arbitrary
    wavelength grid in order to measure the continuum luminosity accurately. 
    The returned continuum contribution is then interpolated from this grid and 
    broadened. 
    """
    f_grid = zeros_like(x_grid, dtype=float64)
    mask = (x_grid <= edge)

    f_grid[mask] += continuum.evaluate(
        x_grid[mask], flux, temp,
        edge=edge, boltz=boltz,
    )
    f_grid[mask] *= attenuation.evaluate(
        x_grid[mask], tau, scale,
        edge=edge,
    )

    cont_lumin = dot(f_grid[mask], x_grid[mask]) * sigma_res
    series_lumin = ratio * cont_lumin

    f_grid = convolve_signal.__wrapped__(f_grid, kernel(fwhm, sigma_res))
    if isinstance(x, ndarray):
        if (x == x_grid).all(): 
            f = f_grid
        elif interpolation_matrix is None:
            f = interp(x, x_grid, f_grid, left=0, right=0)
        else:
            M, b = interpolation_matrix
            f = M @ f_grid + b
    else:
        f = interp(x, x_grid, f_grid, left=0, right=0)

    f += series.evaluate(
        x, series_lumin, fwhm,
        sigma_res=sigma_res, waves=waves, weights=weights,
    )

    return f

def fit_deriv(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    temp: float,
    tau: float,
    scale: float,
    ratio: float,
    *,
    sigma_res: float = None,
    edge: float = None,
    waves: NDArray[float64] = None,
    weights: NDArray[float64] = None,
    boltz: float = None,
    x_grid: NDArray[float64] = None,
    fixed: dict[str, bool] | None = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> list[NDArray[float64]]:
    
    df_dflux:  ndarray[float] = zeros_like(x, dtype=float)
    df_dfwhm:  ndarray[float] = zeros_like(x, dtype=float)
    df_dtemp:  ndarray[float] = zeros_like(x, dtype=float)
    df_dtau:   ndarray[float] = zeros_like(x, dtype=float)
    df_dscale: ndarray[float] = zeros_like(x, dtype=float)
    df_dratio: ndarray[float] = zeros_like(x, dtype=float)

    if all(fixed.values()):
        return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale, df_dratio]
    
    if (x == x_grid).all():
        transform = lambda y: y
    elif interpolation_matrix is not None:
        M, b = interpolation_matrix
        transform = lambda y: M @ y + b
    else:
        transform = lambda y: interp(x, x_grid, y, left=0, right=0)
    
    _k: ndarray[float] = None
    def k() -> ndarray[float]:
        nonlocal _k
        if _k is None: _k = kernel(fwhm, sigma_res)
        return _k
    
    _dk: list[ndarray[float]] = None
    def dk() -> ndarray[float]: 
        nonlocal _dk
        if _dk is None: _dk = kernel_deriv(fwhm, sigma_res)
        return _dk

    _cont_grid: ndarray[float] = None
    def cont_grid() -> ndarray[float]:
        nonlocal _cont_grid
        if _cont_grid is None:
            _cont_grid = continuum.evaluate(
                x_grid, flux, temp,
                edge=edge, boltz=boltz,
            )
        return _cont_grid

    _cont: ndarray[float] = None
    def cont() -> ndarray[float]:
        nonlocal _cont
        if _cont is None:
            _cont = transform(cont_grid())
        return _cont        
    
    _dcont: list[ndarray[float]] = None
    def dcont() -> list[ndarray[float]]:
        nonlocal _dcont
        if _dcont is None:
            _dcont = continuum.fit_deriv(
                x, flux, temp,
                edge=edge, boltz=boltz,
                fixed={'flux': fixed['flux'], 'temp': fixed['temp']},
            )
        return _dcont

    _atte_grid: ndarray[float] = None
    def atte_grid() -> ndarray[float]:
        nonlocal _atte_grid
        if _atte_grid is None:
            _atte_grid = attenuation.evaluate(
                x_grid, tau, scale,
                edge=edge,
            )
        return _atte_grid    

    _atte: ndarray[float] = None
    def atte() -> ndarray[float]:
        nonlocal _atte
        if _atte is None:
            _atte = transform(atte_grid())
        return _atte    

    _datte: list[ndarray[float]] = None
    def datte() -> list[ndarray[float]]:
        nonlocal _datte
        if _datte is None:
            _datte = attenuation.fit_deriv(
                x, tau, scale,
                edge=edge,
                fixed={'tau': fixed['tau'], 'scale': fixed['scale']},
            )
        return _datte

    _cont_lumin: float = None
    def cont_lumin() -> float:
        nonlocal _cont_lumin
        if _cont_lumin is None:
            f_grid: ndarray[float] = cont_grid() * atte_grid()
            _cont_lumin = dot(f_grid, x_grid) * sigma_res
        return _cont_lumin 

    _seri_lumin: float = None
    def seri_lumin() -> float:
        nonlocal _seri_lumin
        if _seri_lumin is None:
            _seri_lumin = ratio * cont_lumin()
        return _seri_lumin    

    _seri: ndarray[float] = None        
    def seri() -> ndarray[float]:
        nonlocal _seri
        if _seri is None:
            _seri = series.evaluate(
                x, seri_lumin(), fwhm,
                sigma_res=sigma_res, waves=waves, weights=weights,
            )
        return _seri

    _dseri: list[ndarray[float]] = None
    def dseri() -> list[ndarray[float]]:
        nonlocal _dseri
        if _dseri is None:
            _dseri = series.fit_deriv(
                x, seri_lumin(), fwhm,
                sigma_res=sigma_res, waves=waves, weights=weights,
                fixed={
                    'lumin': fixed['flux'] and fixed['ratio'], 
                    'fwhm': fixed['fwhm'],
                },
            )
        return _dseri    

    # w/o pydantic validation
    _convolve_signal = convolve_signal.__wrapped__
    
    if not fixed['flux']:
        df_dflux[:] = _convolve_signal(cont() * atte(), k()) + seri()
        df_dflux[:] /= flux

    if not fixed['fwhm']:
        df_dfwhm[:] = _convolve_signal(cont() * atte(), dk()) + dseri()[1]

    if not fixed['temp']:
        df_dtemp[:] = _convolve_signal(dcont()[1] * atte(), k())

    if not fixed['tau']:
        df_dtau[:] = _convolve_signal(cont() * datte()[0], k())

    if not fixed['scale']:
        df_dscale[:] = _convolve_signal(cont() * datte()[1], k())

    if not fixed['ratio']:
        df_dratio[:] = dseri()[0] * cont_lumin()

    return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale, df_dratio]

def evaluate_template(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: Template = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> float | NDArray[float64]:
    """
    Evaluates the Balmer pseudo-continuum model using linear interpolation with
    a BalmerTemplate instance.
    """
    idx = _identify_closest_idx(template.fwhm, fwhm)

    if template.fwhm[idx] == fwhm:
        y = template.data[idx]
    else:
        x0, x1 = template.fwhm[idx:idx+2]
        y0, y1 = template.data[idx:idx+2]
        y = y0 + (y1 - y0) / (x1 - x0) * (fwhm - x0)

    if isinstance(x, ndarray) and (x == template.x).all():
        return flux * y
    elif isinstance(x, ndarray) and not (interpolation_matrix is None):
        M, b = interpolation_matrix
        return flux * (M @ y + b)
    else:
        return flux * interp(x, template.x, y, left=0, right=0)

def fit_deriv_template(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    fixed: dict[str, bool] | None = None,
    template: Template = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> list[ndarray[float]]:

    df_dflux:  ndarray[float] = zeros_like(x, dtype=float)
    df_dfwhm:  ndarray[float] = zeros_like(x, dtype=float)

    # Note: other derivatives are zero by definition
    df_dtemp:  ndarray[float] = zeros_like(x, dtype=float)
    df_dtau:   ndarray[float] = zeros_like(x, dtype=float)
    df_dscale: ndarray[float] = zeros_like(x, dtype=float)
    df_dratio: ndarray[float] = zeros_like(x, dtype=float)

    if not (fixed is None) and not all(fixed.values()):
        idx = _identify_closest_idx(template.fwhm, fwhm)

        x0, x1 = template.fwhm[idx:idx+2]
        y0, y1 = flux * template.data[idx:idx+2]

        grad = (y1 - y0) / (x1 - x0)
        y = y0 + grad * (fwhm - x0)

        if not fixed['flux']:
            if (x == template.x).all():
                df_dflux[:] = y / flux
            elif interpolation_matrix is not None:
                M, b = interpolation_matrix
                df_dflux[:] = (M @ y + b) / flux
            else:
                df_dflux[:] = interp(x, template.x, y, left=0, right=0) / flux

        if not fixed['fwhm']:
            df_dfwhm[:] = grad

    return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale, df_dratio]