from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

cdef void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)

cdef void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)

cdef void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)

### By INTERPOLATION

cdef void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)

cdef void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)

cdef void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)

### By RESCALING

cdef void _fit_deriv_rescale_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
)