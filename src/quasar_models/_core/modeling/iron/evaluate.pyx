from numpy import (
    zeros as np_zeros,
    float64 as np_float64,
)
from quasar_models._core.utils cimport (
    arr_multiply_inplace,
)
from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact,
    _evaluate_interp as _template_evaluate_interp,
)
from quasar_models._core.modeling.split.evaluate cimport (
    _evaluate as _split_evaluate,
)

### By CONVOLUTION

cdef inline void _evaluate_exact_no_split(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    _template_evaluate_exact(
        y,
        flux, fwhm, 
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def evaluate_exact_no_split(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    _evaluate_exact_no_split(
        y, 
        flux, fwhm, split, left, right, 
        template_fwhm, template_x, template_data, 
        sigma_res, scale, n_scales,
    )

cdef inline void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, template_x.shape[0]), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])

    cdef const double[::1] modified_fwhm = template_fwhm[0:1]

    _template_evaluate_exact(
        y,
        flux, fwhm, 
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )

def evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    _evaluate_exact(
        y, 
        flux, fwhm, split, left, right, 
        template_fwhm, template_x, template_data, 
        sigma_res, scale, n_scales,
    )

### By INTERPOLATION // No split possible

cdef inline void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    _template_evaluate_interp(
        y,
        flux, fwhm, 
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    _evaluate_interp(
        y, 
        flux, fwhm, split, left, right, 
        template_fwhm, template_x, template_data, 
        sigma_res, scale, n_scales,
    )