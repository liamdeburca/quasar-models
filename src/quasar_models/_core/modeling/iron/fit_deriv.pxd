from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

### By CONVOLUTION // No split

cdef void _fit_deriv_exact_no_split_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_no_split_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_no_split_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

### By CONVOLUTION // With split

# Special cases: one free parameter

cdef void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_only_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_only_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_only_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

## Special cases: two free parameters

# Flux and ...

cdef void _fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

# FWHM and ...

cdef void _fit_deriv_exact_fwhm_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_fwhm_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_fwhm_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

# Split and ...

cdef void _fit_deriv_exact_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

# Left and ...

cdef void _fit_deriv_exact_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

## Special cases: three free parameters

# Flux and ...
cdef void _fit_deriv_exact_flux_and_fwhm_and_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_fwhm_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_fwhm_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_flux_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

# FWHM and ...

cdef void _fit_deriv_exact_fwhm_and_split_and_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_fwhm_and_split_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_fwhm_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

# Split and ...

cdef void _fit_deriv_exact_split_and_left_and_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

## Special cases: four free parameters

cdef void _fit_deriv_exact_all_except_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_all_except_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_all_except_split(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_all_except_left(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_exact_all_except_right(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

## General case: all free parameters

cdef void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

### By INTERPOLATION

cdef void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)
