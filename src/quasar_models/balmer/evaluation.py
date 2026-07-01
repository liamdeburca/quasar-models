from typing import Protocol
from numpy import zeros_like, float64
from .continuum import BalmerContinuumTemplate
from .series import BalmerSeriesTemplate

from quasar_core.modelling.template import evaluate_exact

from ..utils.template import evaluation as template_evaluation

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

class EvaluateFunc(Protocol):
    def __call__(
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
        ...

class FitDerivFunc(Protocol):
    def __call__(
        self,
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
    ) -> list[FloatVector]:
        ...

# By CONVOLUTION

def evaluate_exact(
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

def fit_deriv_exact(
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
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False, 'ratio': False}

    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)
    df_dratio = zeros_like(x, dtype=float64)

    if not all(fixed.values()):
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
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False, 'ratio': False}

    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)
    df_dratio = zeros_like(x, dtype=float64)

    if not all(fixed.values()):
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

###

def choose_evaluate_func(
    perform_interp_fitting: bool,
) -> EvaluateFunc:
    return evaluate_interp if perform_interp_fitting else evaluate_exact

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
    perform_interp_fitting: bool | None = None,
    evaluate_func: EvaluateFunc | None = None,
) -> FloatVector:
    if evaluate_func is None:
        assert perform_interp_fitting is not None
        evaluate_func = choose_evaluate_func(perform_interp_fitting)

    return evaluate_func(
        x, flux, fwhm, ratio,
        continuum_template=continuum_template,
        series_template=series_template,
        continuum_interpolation_matrix=continuum_interpolation_matrix,
        series_interpolation_matrix=series_interpolation_matrix,
    )

def choose_fit_deriv_func(
    fixed: dict[str, bool] | None,
    perform_interp_fitting: bool,
) -> FitDerivFunc:
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False, 'ratio': False}

    return fit_deriv_interp if perform_interp_fitting else fit_deriv_exact

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    ratio: float,
    *,
    continuum_template: BalmerContinuumTemplate,
    series_template: BalmerSeriesTemplate,
    perform_interp_fitting: bool | None = None,
    continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
    fit_deriv_func: FitDerivFunc | None = None,
) -> list[FloatVector, FloatVector, FloatVector]:
    if fit_deriv_func is None:
        assert perform_interp_fitting is not None
        fit_deriv_func = choose_fit_deriv_func(fixed, perform_interp_fitting)

    return fit_deriv_func(
        x, flux, fwhm, ratio,
        continuum_template=continuum_template,
        series_template=series_template,
        continuum_interpolation_matrix=continuum_interpolation_matrix,
        series_interpolation_matrix=series_interpolation_matrix,
        fixed=fixed,
    )