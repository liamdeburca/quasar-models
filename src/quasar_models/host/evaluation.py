from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

from .host_galaxy_template import HostGalaxyTemplate
from ..utils.template import evaluation

def evaluate(
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

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    host_galaxy_template: HostGalaxyTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    return evaluation.fit_deriv(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
        fixed=fixed,
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
    return evaluation.fit_deriv_interp(
        x, flux, fwhm, 
        template=host_galaxy_template,
        interpolation_matrix=interpolation_matrix,
        fixed=fixed,
    )