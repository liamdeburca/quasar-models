from libc.math cimport (
    sqrt as math_sqrt,
)
from numpy import (
    zeros as np_zeros,
    float64 as np_float64,
)
from quasar_models._core.convolution.utils cimport (
    _kernel,
    _convolve_signal,
    _convolve_signal2d,
)
from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact  as _template_evaluate_exact,
    _evaluate_interp as _template_evaluate_interp,
)
from quasar_models._core.modeling.template.fit_deriv cimport (
    _fit_deriv_exact_only_flux as _template_fit_deriv_exact_only_flux,
    _fit_deriv_exact_only_fwhm as _template_fit_deriv_exact_only_fwhm,
    _fit_deriv_exact_all       as _template_fit_deriv_exact_all,

    _fit_deriv_interp_only_flux as _template_fit_deriv_interp_only_flux,
    _fit_deriv_interp_only_fwhm as _template_fit_deriv_interp_only_fwhm,
    _fit_deriv_interp_all       as _template_fit_deriv_interp_all,
)
from quasar_models._core.modeling.iron.evaluate cimport (
    _evaluate_exact,
)
from quasar_models._core.modeling.split.fit_deriv cimport (
    _fit_deriv_only_split      as _split_fit_deriv_only_split,
    _fit_deriv_only_left       as _split_fit_deriv_only_left,
    _fit_deriv_only_right      as _split_fit_deriv_only_right,
    _fit_deriv_split_and_left  as _split_fit_deriv_split_and_left,
    _fit_deriv_split_and_right as _split_fit_deriv_split_and_right,
    _fit_deriv_left_and_right  as _split_fit_deriv_left_and_right,
    _fit_deriv_all             as _split_fit_deriv_all,
)
from quasar_models._core.utils cimport (
    arr_multiply_inplace,
    arr_add_inplace,
    multiply_and_multiply_to,
)

from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

### By CONVOLUTION // No split

cdef inline void _fit_deriv_exact_no_split_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        cytemplate,
        n_scales,
    )

def fit_deriv_exact_no_split_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_no_split_only_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

cdef inline void _fit_deriv_exact_no_split_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        cytemplate,
        n_scales,
    )

def fit_deriv_exact_no_split_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_no_split_only_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

cdef inline void _fit_deriv_exact_no_split_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        cytemplate,
        n_scales,
    )

def fit_deriv_exact_no_split_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_no_split_all(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

### By CONVOLUTION // With split

# Special cases: one free parameter

cdef inline void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
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
        derivs[0, :],
        1.0, fwhm, split, left, right,
        cytemplate,
        scale, n_scales,
    )

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)

    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm, 
        split_template,
        n_scales,
    )

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    cytemplate.add_split_fit_deriv_only_split(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_only_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_only_split(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    cytemplate.add_split_fit_deriv_only_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_only_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_only_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales
):
    cytemplate.add_split_fit_deriv_only_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_only_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_only_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate,
        scale, n_scales,
    )

## Special cases: two free parameters

# Flux and ...

cdef inline void _fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux and fwhm derivatives with split-weighted template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )

def fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative with split-weighted template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_split(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_split(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative with split-weighted template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative with split-weighted template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

# FWHM and ...

cdef inline void _fit_deriv_exact_fwhm_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative with split-weighted template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_split(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_fwhm_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_split(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative with split-weighted template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_fwhm_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    # Create split-weighted template
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative with split-weighted template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template, n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_fwhm_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

# Split and ...

cdef inline void _fit_deriv_exact_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    cytemplate.add_split_fit_deriv_split_and_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_split_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    cytemplate.add_split_fit_deriv_split_and_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_split_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

# Left and ...

cdef inline void _fit_deriv_exact_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    
    cytemplate.add_split_fit_deriv_left_and_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

## Special cases: three free parameters

# Flux and ...
cdef inline void _fit_deriv_exact_flux_and_fwhm_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
        
    # Compute flux and fwhm derivatives with split template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_split(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_fwhm_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_fwhm_and_split(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_fwhm_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux and fwhm derivatives with split template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_fwhm_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_fwhm_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )
cdef inline void _fit_deriv_exact_flux_and_fwhm_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux and fwhm derivatives with split template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    cytemplate.add_split_fit_deriv_only_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_fwhm_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_fwhm_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )
cdef inline void _fit_deriv_exact_flux_and_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative with split template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        scale,
    )
    
    cytemplate.add_split_fit_deriv_split_and_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_split_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative with split template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        scale,
    )
    
    cytemplate.add_split_fit_deriv_split_and_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_split_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative with split template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        scale,
    )
    
    cytemplate.add_split_fit_deriv_left_and_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_flux_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

# FWHM and ...

cdef inline void _fit_deriv_exact_fwhm_and_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative with split template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        scale,
    )
    
    cytemplate.add_split_fit_deriv_split_and_left(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_fwhm_and_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_split_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative with split template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        scale,
    )
    
    cytemplate.add_split_fit_deriv_split_and_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_fwhm_and_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_split_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative with split template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        scale,
    )
    
    cytemplate.add_split_fit_deriv_left_and_right(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_fwhm_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

# Split and ...

cdef inline void _fit_deriv_exact_split_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    cytemplate.add_split_fit_deriv_all(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_split_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_split_and_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

## Special cases: four free parameters

cdef inline void _fit_deriv_exact_all_except_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute fwhm derivative only with split template
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    cytemplate.add_split_fit_deriv_all(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_all_except_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_all_except_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux derivative only with split template
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    cytemplate.add_split_fit_deriv_all(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_all_except_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_all_except_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
        
    # Compute flux and fwhm derivatives only (split is not a free parameter)
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )

    if flux != 0.0:
        cytemplate.add_split_fit_deriv_left_and_right(
            derivs[2:5, :],
            flux, fwhm, split, left, right,
            scale, n_scales,
        )

def fit_deriv_exact_all_except_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_all_except_split(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux and fwhm derivatives with split template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    if flux != 0.0:
        cytemplate.add_split_fit_deriv_split_and_right(
            derivs[2:5, :],
            flux, fwhm, split, left, right,
            scale, n_scales,
        )

def fit_deriv_exact_all_except_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_all_except_left(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux and fwhm derivatives with split template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    
    if flux != 0.0:
        cytemplate.add_split_fit_deriv_split_and_left(
            derivs[2:5, :],
            flux, fwhm, split, left, right,
            scale, n_scales,
        )

def fit_deriv_exact_all_except_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_all_except_right(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

## General case: all free parameters

cdef inline void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_init, fwhm_kernel
    cdef Py_ssize_t n = cytemplate.x.shape[0]
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Compute flux and fwhm derivatives with split template
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        split_template,
        n_scales,
    )
    cytemplate.add_split_fit_deriv_all(
        derivs[2:5, :],
        flux, fwhm, split, left, right,
        scale, n_scales,
    )

def fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_exact_all(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

### By INTERPOLATION

cdef inline void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_fit_deriv_interp_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        cytemplate, n_scales,
    )


def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_fit_deriv_interp_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )

cdef inline void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _template_fit_deriv_interp_all(
        derivs[0:2, :],
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
):
    _fit_deriv_interp_all(
        derivs,
        flux, fwhm,
        split, left, right,
        cytemplate, scale, n_scales,
    )