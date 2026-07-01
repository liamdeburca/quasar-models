from typing import Protocol
from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_typing.scipy import csr_matrix_

from .host_galaxy_template import HostGalaxyTemplate
from ..utils.template import evaluation

class EvaluateFunc(Protocol):
    def __call__(
        self,
        x: FloatVector, 
        flux: float, 
        fwhm: float, 
        *,
        host_galaxy_template: HostGalaxyTemplate,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> FloatVector:
        ...

class FitDerivFunc(Protocol):
    def __call__(
        self,
        x: FloatVector, 
        flux: float, 
        fwhm: float, 
        *,
        template: HostGalaxyTemplate | None = None,
        template_fwhm: FloatVector | None = None,
        template_x: FloatVector | None = None,
        template_data: FloatMatrix | None = None,
        sigma_res: float | None = None,
        normalisation: float | None = None,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        fixed: dict[str, bool] | None = None,
    ) -> list[FloatVector]:
        ...

### By CONVOLUTION

def evaluate_exact(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector: 
    return evaluation.evaluate(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
    )

def fit_deriv_exact(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False}
    return evaluation.fit_deriv(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
        fixed=fixed,
    )

### By INTERPOLATION

def evaluate_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    return evaluation.evaluate_interp(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
    )

def fit_deriv_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False}
    return evaluation.fit_deriv_interp(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
        fixed=fixed,
    )

###

def choose_evaluate_func(
    perform_interp_fitting: bool,
) -> EvaluateFunc:
    return evaluate_interp if perform_interp_fitting else evaluate_exact

def evaluate(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    perform_interp_fitting: bool | None = None,
    evaluate_func: EvaluateFunc | None = None,
) -> FloatVector:
    if evaluate_func is None:
        assert perform_interp_fitting is not None
        evaluate_func = choose_evaluate_func(perform_interp_fitting)

    return evaluate_func(
        x, flux, fwhm, 
        host_galaxy_template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
    )

def choose_fit_deriv_func(
    fixed: dict[str, bool] | None,
    perform_interp_fitting: bool,
) -> FitDerivFunc:
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False}

    return fit_deriv_interp if perform_interp_fitting else fit_deriv_exact

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
    perform_interp_fitting: bool | None = None,
    fit_deriv_func: FitDerivFunc | None = None,
) -> list[FloatVector]:
    if fixed is None:
        fixed = {'flux': False, 'fwhm': False}

    if fit_deriv_func is None:
        assert perform_interp_fitting is not None
        fit_deriv_func = choose_fit_deriv_func(fixed, perform_interp_fitting)
    
    return fit_deriv_func(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
        fixed=fixed,
    )