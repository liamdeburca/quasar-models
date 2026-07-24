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
from quasar_models._core.modeling.split.evaluate cimport (
    _evaluate as _split_evaluate,
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

### By CONVOLUTION // No split

cdef inline void _fit_deriv_exact_no_split_only_flux(
    double[:,::1] derivs,
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
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_no_split_only_flux(
    double[:,::1] derivs,
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
    _fit_deriv_exact_no_split_only_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_no_split_only_fwhm(
    double[:,::1] derivs,
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
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_no_split_only_fwhm(
    double[:,::1] derivs,
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
    _fit_deriv_exact_no_split_only_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_no_split_all(
    double[:,::1] derivs,
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
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_no_split_all(
    double[:,::1] derivs,
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
    _fit_deriv_exact_no_split_all(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
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
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    _evaluate_exact(
        derivs[0, :],
        1.0, fwhm, split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
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
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
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
    cdef double[:,::1] modified_data = np_zeros((1, template_x.size), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])

    cdef const double[::1] modified_fwhm = template_fwhm[0:1]

    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm, 
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
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
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_split(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]    
    if flux != 0.0:
        # Compute split derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_split(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )

        # Only split derivative
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])

def fit_deriv_exact_only_split(
    double[:,::1] derivs,
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
    _fit_deriv_exact_only_split(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    if flux != 0.0:
        # Compute left derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )

        # Only left derivative
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_only_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_only_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_only_right(
    double[:,::1] derivs,
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
    const double n_scales
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    if flux != 0.0:
        # Compute right derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )

        # Only right derivative
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_only_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_only_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
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
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    cdef Py_ssize_t n = template_x.shape[0]

    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_split(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute split derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_split(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split derivative
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])

def fit_deriv_exact_flux_and_split(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_split(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute left derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left derivative
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)        
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_flux_and_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute right derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only right derivative
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_flux_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

# FWHM and ...

cdef inline void _fit_deriv_exact_fwhm_and_split(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute fwhm derivative with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute split derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_split(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split derivative
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])

def fit_deriv_exact_fwhm_and_split(
    double[:,::1] derivs,
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
    _fit_deriv_exact_fwhm_and_split(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute fwhm derivative with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute left derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left derivative
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_fwhm_and_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_fwhm_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute fwhm derivative with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute right derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only right derivative
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_fwhm_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_fwhm_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

# Split and ...

cdef inline void _fit_deriv_exact_split_and_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    if flux != 0.0:
        # Compute split and left derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and left derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)       
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_split_and_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_split_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_split_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    if flux != 0.0:
        # Compute split and right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and right derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_split_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_split_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

# Left and ...

cdef inline void _fit_deriv_exact_left_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    if flux != 0.0:
        # Compute left and right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_left_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left and right derivatives
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_left_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
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
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double scale,
    const double n_scales,
):
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
        
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute split derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_split(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split derivative
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])

def fit_deriv_exact_flux_and_fwhm_and_split(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_fwhm_and_split(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_fwhm_and_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute left derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left derivative
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_flux_and_fwhm_and_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_fwhm_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )
cdef inline void _fit_deriv_exact_flux_and_fwhm_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute right derivative only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_only_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only right derivative
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_flux_and_fwhm_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_fwhm_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )
cdef inline void _fit_deriv_exact_flux_and_split_and_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, scale,
    )
    
    if flux != 0.0:
        # Compute split and left derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and left derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_flux_and_split_and_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_split_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_split_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, scale,
    )
    
    if flux != 0.0:
        # Compute split and right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and right derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_flux_and_split_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_split_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_left_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, scale,
    )
    
    if flux != 0.0:
        # Compute left and right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_left_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left and right derivatives
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_flux_and_left_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_flux_and_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

# FWHM and ...

cdef inline void _fit_deriv_exact_fwhm_and_split_and_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    
    # Compute fwhm derivative with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, modified_data,
        sigma_res, scale,
    )
    
    if flux != 0.0:
        # Compute split and left derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and left derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_fwhm_and_split_and_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_fwhm_and_split_and_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_split_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    
    # Compute fwhm derivative with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, modified_data,
        sigma_res, scale,
    )
    
    if flux != 0.0:
        # Compute split and right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and right derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_fwhm_and_split_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_fwhm_and_split_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_left_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    
    # Compute fwhm derivative with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, modified_data,
        sigma_res, scale,
    )
    
    if flux != 0.0:
        # Compute left and right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_left_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left and right derivatives
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_fwhm_and_left_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_fwhm_and_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

# Split and ...

cdef inline void _fit_deriv_exact_split_and_left_and_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    if flux != 0.0:
        # Compute all three split derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_all(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only all split, left, and right derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_split_and_left_and_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_split_and_left_and_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

## Special cases: four free parameters

cdef inline void _fit_deriv_exact_all_except_flux(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute fwhm derivative only with modified template data
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute all three split derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_all(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # All split derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_all_except_flux(
    double[:,::1] derivs,
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
    _fit_deriv_exact_all_except_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_fwhm(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux derivative only with modified template data
    _template_fit_deriv_exact_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute all three split derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_all(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # All split derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_all_except_fwhm(
    double[:,::1] derivs,
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
    _fit_deriv_exact_all_except_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_split(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
        
    # Compute flux and fwhm derivatives only (split is not a free parameter)
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )

    if flux != 0.0:
        # Compute all three split derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_all(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only left and right derivatives
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_all_except_split(
    double[:,::1] derivs,
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
    _fit_deriv_exact_all_except_split(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_left(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute split and right derivatives only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_right(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and right derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_all_except_left(
    double[:,::1] derivs,
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
    _fit_deriv_exact_all_except_left(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_exact_all_except_right(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_kernel, fwhm_init
    cdef Py_ssize_t n = template_x.shape[0]
    
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    
    if flux != 0.0:
        # Compute split and left derivatives only
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_split_and_left(
            split_derivs,
            template_x,
            split, left, right,
            sigma_res, scale,
        )
        
        # Only split and left derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])

def fit_deriv_exact_all_except_right(
    double[:,::1] derivs,
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
    _fit_deriv_exact_all_except_right(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

## General case: all free parameters

cdef inline void _fit_deriv_exact_all(
    double[:,::1] derivs,
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
    # Declare variables at start
    cdef double[:,::1] split_derivs
    cdef double[::1] kernel
    cdef double fwhm_init, fwhm_kernel
    cdef Py_ssize_t n = template_x.shape[0]
        
    # Create modified template data
    cdef double[:,::1] modified_data = np_zeros((1, n), dtype=np_float64)
    _split_evaluate(modified_data[0, :], template_x, split, left, right, sigma_res, scale)
    arr_multiply_inplace(modified_data[0, :], template_data[0, :])
    
    # Create modified template fwhm
    cdef const double[::1] modified_fwhm = template_fwhm[0:1]
    
    # Compute flux and fwhm derivatives with modified template data
    _template_fit_deriv_exact_all(
        derivs[0:2, :],
        flux, fwhm,
        modified_fwhm, template_x, modified_data,
        sigma_res, n_scales,
    )
    if flux != 0.0:
        # Compute split, left, right derivatives
        split_derivs = np_zeros((3, n), dtype=np_float64)
        _split_fit_deriv_all(
            split_derivs, 
            template_x, 
            split, left, right, 
            sigma_res, scale,
        )

        # All split, left, right derivatives
        multiply_and_multiply_to(split_derivs[0, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[1, :], template_data[0, :], flux)
        multiply_and_multiply_to(split_derivs[2, :], template_data[0, :], flux)
        # Get initial fwhm
        fwhm_init = template_fwhm[0]
        
        if fwhm_init != fwhm:
            # Perform convolution
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        arr_add_inplace(derivs[2, :], split_derivs[0, :])
        arr_add_inplace(derivs[3, :], split_derivs[1, :])
        arr_add_inplace(derivs[4, :], split_derivs[2, :])

def fit_deriv_exact_all(
    double[:,::1] derivs,
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
    _fit_deriv_exact_all(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

### By INTERPOLATION

cdef inline void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
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
    _template_fit_deriv_interp_only_flux(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )


def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
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
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
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
    _template_fit_deriv_interp_only_fwhm(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
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
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )

cdef inline void _fit_deriv_interp_all(
    double[:,::1] derivs,
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
    _template_fit_deriv_interp_all(
        derivs[0:2, :],
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
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
    _fit_deriv_interp_all(
        derivs,
        flux, fwhm,
        split, left, right,
        template_fwhm, template_x, template_data,
        sigma_res, scale, n_scales,
    )