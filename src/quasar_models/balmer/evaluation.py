from numpy import zeros_like, float64
from .continuum import BalmerContinuumTemplate
from .series import BalmerSeriesTemplate

from ..utils.template import evaluation as template_evaluation

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

# By CONVOLUTION

def evaluate(
    x: FloatVector,
    flux: float,
    fwhm: float,
    ratio: float,
    *,
    continuum_template: BalmerContinuumTemplate,
    series_template: BalmerSeriesTemplate,
    continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    f_cont = template_evaluation.evaluate(
        x, flux, fwhm,
        template=continuum_template,
        interpolation_matrix=continuum_interpolation_matrix,
    )
    f_series = template_evaluation.evaluate(
        x, flux, fwhm,
        template=series_template,
        interpolation_matrix=series_interpolation_matrix,
    )
    return f_cont + ratio * f_series

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    ratio: float,
    *,
    continuum_template: BalmerContinuumTemplate,
    series_template: BalmerSeriesTemplate,
    continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector, FloatVector, FloatVector]:
    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)
    df_dratio = zeros_like(x, dtype=float64)

    if (fixed is not None) and not all(fixed.values()):
        if (not fixed['flux'] and ratio != 0) \
            or (not fixed['ratio'] and flux != 0):
            f_series = template_evaluation.evaluate(
                x, 1.0, fwhm,
                template=series_template,
                interpolation_matrix=series_interpolation_matrix,
            )
        
        if not fixed['flux']:
            df_dflux[:] = template_evaluation.evaluate(
                    x, 1.0, fwhm,
                    template=continuum_template,
                    interpolation_matrix=continuum_interpolation_matrix,
                )
            if ratio != 0:
                df_dflux[:] += ratio * f_series

        if not fixed['fwhm'] and flux != 0:
            df_dfwhm[:] = template_evaluation.fit_deriv(
                x, 1.0, fwhm,
                template=continuum_template,
                interpolation_matrix=continuum_interpolation_matrix,
                fixed={'flux': True, 'fwhm': False},
            )[1]
            if ratio != 0:
                df_dfwhm[:] += ratio * template_evaluation.fit_deriv(
                    x, 1.0, fwhm,
                    template=series_template,
                    interpolation_matrix=series_interpolation_matrix,
                    fixed={'flux': True, 'fwhm': False},
                )[1]
            
            df_dfwhm[:] *= flux

        if not fixed['ratio'] and flux != 0:
            df_dratio[:] = flux * f_series

    return [df_dflux, df_dfwhm, df_dratio]

# By INTERPOLATION

def evaluate_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    ratio: float,
    *,
    continuum_template: BalmerContinuumTemplate,
    series_template: BalmerSeriesTemplate,
    continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    f_cont = template_evaluation.evaluate_interp(
        x, flux, fwhm,
        template=continuum_template,
        interpolation_matrix=continuum_interpolation_matrix,
    )
    f_series = template_evaluation.evaluate_interp(
        x, flux, fwhm,
        template=series_template,
        interpolation_matrix=series_interpolation_matrix,
    )
    return f_cont + ratio * f_series

def fit_deriv_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    ratio: float,
    *,
    continuum_template: BalmerContinuumTemplate,
    series_template: BalmerSeriesTemplate,
    continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector, FloatVector, FloatVector]:
    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)
    df_dratio = zeros_like(x, dtype=float64)

    if (fixed is not None) and not all(fixed.values()):
        if not fixed['flux'] or (not fixed['ratio'] and flux != 0):
            f_series = template_evaluation.evaluate_interp(
                x, 1.0, fwhm,
                template=series_template,
                interpolation_matrix=series_interpolation_matrix,
            )
        
        if not fixed['flux']:
            f_continuum = template_evaluation.evaluate_interp(
                    x, 1.0, fwhm,
                    template=continuum_template,
                    interpolation_matrix=continuum_interpolation_matrix,
                )
            df_dflux[:] = f_continuum + ratio * f_series

        if not fixed['fwhm'] and flux != 0:
            df_dfwhm[:] = template_evaluation.fit_deriv_interp(
                x, 1.0, fwhm,
                template=continuum_template,
                interpolation_matrix=continuum_interpolation_matrix,
                fixed={'flux': True, 'fwhm': False},
            )[1]
            if ratio != 0:
                df_dfwhm[:] += ratio * template_evaluation.fit_deriv_interp(
                    x, 1.0, fwhm,
                    template=series_template,
                    interpolation_matrix=series_interpolation_matrix,
                    fixed={'flux': True, 'fwhm': False},
                )[1]
            df_dfwhm[:] *= flux

        if not fixed['ratio'] and flux != 0:
            df_dratio[:] = flux * f_series

    return [df_dflux, df_dfwhm, df_dratio]

# def evaluate_sparse(
#     x: NDArray[float64],
#     flux: float,
#     fwhm: float,
#     temp: float,
#     tau: float,
#     scale: float,
#     ratio: float,
#     *,
#     sigma_res: float = None,
#     edge: float = None,
#     waves: NDArray[float64] = None,
#     weights: NDArray[float64] = None,
#     boltz: float = None,
#     x_grid: NDArray[float64] = None,
#     interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
# ) -> tuple[NDArray[bool_], NDArray[float64]]:
#     return (
#         ones_like(x, dtype=bool_),
#         evaluate(
#             x,
#             flux, fwhm, temp, tau, scale, ratio,
#             sigma_res=sigma_res, 
#             edge=edge, 
#             waves=waves, 
#             weights=weights, 
#             boltz=boltz,
#             x_grid=x_grid,
#             interpolation_matrix=interpolation_matrix,
#         )
#     )

# def evaluate_interp_sparse(
#     x: float | NDArray[float64],
#     flux: float,
#     fwhm: float,
#     *,
#     template: BaseTemplate = None,
#     interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,   
# ) -> tuple[NDArray[bool_], NDArray[float64]]:
#     return (
#         ones_like(x, dtype=bool_),
#         evaluate_interp(
#             x, flux, fwhm,
#             template=template,
#             interpolation_matrix=interpolation_matrix,
#         )
#     )

# def fit_deriv(
#     x: NDArray[float64],
#     flux: float,
#     fwhm: float,
#     temp: float,
#     tau: float,
#     scale: float,
#     ratio: float,
#     *,
#     sigma_res: float = None,
#     edge: float = None,
#     waves: NDArray[float64] = None,
#     weights: NDArray[float64] = None,
#     boltz: float = None,
#     x_grid: NDArray[float64] = None,
#     fixed: dict[str, bool] | None = None,
#     interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
# ) -> list[NDArray[float64]]:
    
#     df_dflux:  ndarray[float] = zeros_like(x, dtype=float)
#     df_dfwhm:  ndarray[float] = zeros_like(x, dtype=float)
#     df_dtemp:  ndarray[float] = zeros_like(x, dtype=float)
#     df_dtau:   ndarray[float] = zeros_like(x, dtype=float)
#     df_dscale: ndarray[float] = zeros_like(x, dtype=float)
#     df_dratio: ndarray[float] = zeros_like(x, dtype=float)

#     if (fixed is None) or all(fixed.values()):
#         return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale, df_dratio]
    
#     #! transform only used to re-bin continuum contrib., but this is analytical?
#     transform = get_template_transform(x, x_grid, interpolation_matrix)
    
#     _k: ndarray[float] = None
#     def k() -> ndarray[float]:
#         nonlocal _k
#         if _k is None: _k = kernel(fwhm, sigma_res)
#         return _k
    
#     _dk: list[ndarray[float]] = None
#     def dk() -> ndarray[float]: 
#         nonlocal _dk
#         if _dk is None: _dk = kernel_deriv(fwhm, sigma_res)
#         return _dk

#     _cont_grid: ndarray[float] = None
#     def cont_grid() -> ndarray[float]:
#         nonlocal _cont_grid
#         if _cont_grid is None:
#             _cont_grid = continuum.evaluate(
#                 x_grid, flux, temp,
#                 edge=edge, boltz=boltz,
#             )
#         return _cont_grid

#     _dcont_grid: list[ndarray[float]] = None
#     def dcont_grid() -> list[ndarray[float]]:
#         nonlocal _dcont_grid
#         if _dcont_grid is None:
#             _dcont_grid = continuum.fit_deriv(
#                 x_grid, flux, temp,
#                 edge=edge, boltz=boltz,
#                 fixed={'flux': fixed['flux'], 'temp': fixed['temp']},
#             )
#         return _dcont_grid

#     _atte_grid: ndarray[float] = None
#     def atte_grid() -> ndarray[float]:
#         nonlocal _atte_grid
#         if _atte_grid is None:
#             _atte_grid = attenuation.evaluate(
#                 x_grid, tau, scale,
#                 edge=edge,
#             )
#         return _atte_grid    

#     _datte_grid: list[ndarray[float]] = None
#     def datte_grid() -> list[ndarray[float]]:
#         nonlocal _datte_grid
#         if _datte_grid is None:
#             _datte_grid = attenuation.fit_deriv(
#                 x_grid, tau, scale,
#                 edge=edge,
#                 fixed={'tau': fixed['tau'], 'scale': fixed['scale']},
#             )
#         return _datte_grid

#     _cont_lumin: float = None
#     def cont_lumin() -> float:
#         nonlocal _cont_lumin
#         if _cont_lumin is None:
#             f_grid: ndarray[float] = cont_grid() * atte_grid()
#             _cont_lumin = dot(f_grid, x_grid) * sigma_res
#         return _cont_lumin 

#     _seri_lumin: float = None
#     def seri_lumin() -> float:
#         nonlocal _seri_lumin
#         if _seri_lumin is None:
#             _seri_lumin = ratio * cont_lumin()
#         return _seri_lumin    

#     _seri: ndarray[float] = None        
#     def seri() -> ndarray[float]:
#         nonlocal _seri
#         if _seri is None:
#             _seri = series.evaluate(
#                 x, seri_lumin(), fwhm,
#                 sigma_res=sigma_res, waves=waves, weights=weights,
#             )
#         return _seri

#     _dseri: list[ndarray[float]] = None
#     def dseri() -> list[ndarray[float]]:
#         nonlocal _dseri
#         if _dseri is None:
#             _dseri = series.fit_deriv(
#                 x, seri_lumin(), fwhm,
#                 sigma_res=sigma_res, waves=waves, weights=weights,
#                 fixed={
#                     'lumin': fixed['flux'] and fixed['ratio'], 
#                     'fwhm': fixed['fwhm'],
#                 },
#             )
#         return _dseri    

#     # w/o pydantic validation
#     _convolve_signal = convolve_signal.__wrapped__
    
#     if not fixed['flux']:
#         df_dflux[:] = _convolve_signal(cont_grid() * atte_grid(), k()) + seri()
#         df_dflux[:] /= flux

#     if not fixed['fwhm']:
#         df_dfwhm[:] = _convolve_signal(cont_grid() * atte_grid(), dk()) + dseri()[1]

#     if not fixed['temp']:
#         df_dtemp[:] = _convolve_signal(dcont_grid()[1] * atte_grid(), k())

#     if not fixed['tau']:
#         df_dtau[:] = _convolve_signal(cont_grid() * datte_grid()[0], k())

#     if not fixed['scale']:
#         df_dscale[:] = _convolve_signal(cont_grid() * datte_grid()[1], k())

#     if not fixed['ratio']:
#         df_dratio[:] = dseri()[0] * cont_lumin()

#     return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale, df_dratio]


# def fit_deriv_interp(
#     x: NDArray[float64],
#     flux: float,
#     fwhm: float,
#     *,
#     fixed: dict[str, bool] | None = None,
#     template: BaseTemplate = None,
#     interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
# ) -> list[ndarray[float]]:

#     df_dflux:  ndarray[float] = zeros_like(x, dtype=float)
#     df_dfwhm:  ndarray[float] = zeros_like(x, dtype=float)

#     # Note: other derivatives are zero by definition
#     df_dtemp:  ndarray[float] = zeros_like(x, dtype=float)
#     df_dtau:   ndarray[float] = zeros_like(x, dtype=float)
#     df_dscale: ndarray[float] = zeros_like(x, dtype=float)
#     df_dratio: ndarray[float] = zeros_like(x, dtype=float)

#     if not (fixed is None) and not all(fixed.values()):
#         transform = get_template_transform(x, template.x, interpolation_matrix)

#         idx = _identify_closest_idx(template.fwhm, fwhm)

#         x0, x1 = template.fwhm[idx:idx+2]
#         y0, y1 = flux * template.data[idx:idx+2]
#         f = y0 + (grad := (y1 - y0) / (x1 - x0)) * (fwhm - x0)

#         if not fixed['flux']: df_dflux[:] = transform(f) / flux
#         if not fixed['fwhm']: df_dfwhm[:] = transform(grad)

#     return [df_dflux, df_dfwhm, df_dtemp, df_dtau, df_dscale, df_dratio]