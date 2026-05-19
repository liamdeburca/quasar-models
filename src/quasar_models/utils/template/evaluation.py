__all__ = [
    'evaluate', 'fit_deriv',
    'evaluate_interp', 'fit_deriv_interp',
]

from numpy import zeros_like, float64

from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_typing.scipy import csr_matrix_

from quasar_utils.convolution import convolve, convolve_deriv, _identify_closest_idx
from ...utils.interpolation import get_template_transform

from .base_template import BaseTemplate

# By CONVOLUTION

def evaluate(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate | None = None,
    template_fwhm: FloatVector | None = None,
    template_x: FloatVector | None = None,
    template_data: FloatMatrix | None = None,
    sigma_res: float | None= None,
    normalisation: float | None = None,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    if flux == 0:
        return zeros_like(x, dtype=float64)
    
    if template_fwhm is None:
        template_fwhm = template.fwhm
    if template_x is None:
        template_x = template.x
    if template_data is None:
        template_data = template.data
    if sigma_res is None:
        sigma_res = template.info.loading.sigma_res
    if normalisation is None:
        normalisation = template.normalisation

    f = convolve.__wrapped__(
        template_data,
        template_fwhm,
        fwhm,
        sigma_res,
    )
    transform = get_template_transform(x, template_x, interpolation_matrix)
    
    return flux * transform(f) / normalisation

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate | None = None,
    template_fwhm: FloatVector | None = None,
    template_x: FloatVector | None = None,
    template_data: FloatMatrix | None = None,
    sigma_res: float | None= None,
    normalisation: float | None = None,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector, FloatVector]:
    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)

    if template_fwhm is None:
        template_fwhm = template.fwhm
    if template_x is None:
        template_x = template.x
    if template_data is None:
        template_data = template.data
    if sigma_res is None:
        sigma_res = template.info.loading.sigma_res
    if normalisation is None:
        normalisation = template.normalisation

    if (fixed is not None) and not all(fixed.values()):
        transform = get_template_transform(x, template_x, interpolation_matrix)

        if not fixed['flux']:
            df_dflux[:] = transform(convolve.__wrapped__(
                template_data,
                template_fwhm,
                fwhm,
                sigma_res,
            )) / normalisation
        if not fixed['fwhm'] and flux != 0:
            df_dfwhm[:] = flux * transform(convolve_deriv.__wrapped__(
                template_data,
                template_fwhm,
                fwhm,
                sigma_res,
            )) / normalisation

    return [df_dflux, df_dfwhm]

# By INTERPOLATION

def evaluate_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate | None = None,
    template_fwhm: FloatVector | None = None,
    template_x: FloatVector | None = None,
    template_data: FloatMatrix | None = None,
    sigma_res: float | None= None,
    normalisation: float | None = None,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    
    if template_fwhm is None:
        template_fwhm = template.fwhm
    if template_x is None:
        template_x = template.x
    if template_data is None:
        template_data = template.data
    if sigma_res is None:
        sigma_res = template.info.loading.sigma_res
    if normalisation is None:
        normalisation = template.normalisation

    idx = _identify_closest_idx(template_fwhm, fwhm)
    if template_fwhm[idx] == fwhm:
        f = template_data[idx]
    else:    
        if idx == template_fwhm.size - 1:
            idx -= 1

        fwhm0, fwhm1 = template_fwhm[idx:idx+2]
        f0, f1 = template_data[idx:idx+2]
        f = f0 + (f1 - f0) * (fwhm - fwhm0) / (fwhm1 - fwhm0)
    
    transform = get_template_transform(x, template_x, interpolation_matrix)

    return flux * transform(f) / normalisation

def fit_deriv_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate | None = None,
    template_fwhm: FloatVector | None = None,
    template_x: FloatVector | None = None,
    template_data: FloatMatrix | None = None,
    sigma_res: float | None= None,
    normalisation: float | None = None,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector, FloatVector]:
    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)

    if template_fwhm is None:
        template_fwhm = template.fwhm
    if template_x is None:
        template_x = template.x
    if template_data is None:
        template_data = template.data
    if sigma_res is None:
        sigma_res = template.info.loading.sigma_res
    if normalisation is None:
        normalisation = template.normalisation

    if (fixed is not None) and not all(fixed.values()):
        transform = get_template_transform(x, template_x, interpolation_matrix)

        idx = _identify_closest_idx(template_fwhm, fwhm, for_deriv=True)
        if idx == template_fwhm.size - 1:
            idx -= 1

        fwhm0, fwhm1 = template_fwhm[idx:idx+2]
        f0, f1 = template_data[idx:idx+2]
        grad = (f1 - f0) / (fwhm1 - fwhm0)

        if not fixed['flux']:
            df_dflux[:] = transform(f0 + grad * (fwhm - fwhm0)) / normalisation

        if not fixed['fwhm'] and flux != 0:
            df_dfwhm[:] = flux * transform(grad) / normalisation

    return [df_dflux, df_dfwhm]