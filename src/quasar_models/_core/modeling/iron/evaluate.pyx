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
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

### By CONVOLUTION

cdef inline void _evaluate_exact_no_split(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_evaluate_exact(
        y,
        flux, fwhm, 
        cytemplate, n_scales,
    )

def evaluate_exact_no_split(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _evaluate_exact_no_split(
        y, 
        flux, fwhm, split, left, right, 
        cytemplate, scale, n_scales,
    )

cdef inline void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate, 
    const double scale,
    const double n_scales,
):
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    _template_evaluate_exact(
        y,
        flux, fwhm,
        split_template, n_scales,
    )

def evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _evaluate_exact(
        y, 
        flux, fwhm, split, left, right, 
        cytemplate, scale, n_scales,
    )

### By INTERPOLATION // No split possible

cdef inline void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_evaluate_interp(
        y,
        flux, fwhm, 
        cytemplate, n_scales,
    )

def evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _evaluate_interp(
        y, 
        flux, fwhm, split, left, right, 
        cytemplate, scale, n_scales,
    )