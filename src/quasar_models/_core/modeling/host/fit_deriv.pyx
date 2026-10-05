from quasar_models._core.modeling.template.fit_deriv cimport (
    _fit_deriv_exact_only_flux as _template_fit_deriv_exact_only_flux,
    _fit_deriv_exact_only_fwhm as _template_fit_deriv_exact_only_fwhm,
    _fit_deriv_exact_all as _template_fit_deriv_exact_all,

    _fit_deriv_interp_only_flux as _template_fit_deriv_interp_only_flux,
    _fit_deriv_interp_only_fwhm as _template_fit_deriv_interp_only_fwhm,
    _fit_deriv_interp_all as _template_fit_deriv_interp_all,

    _fit_deriv_rescale_only_flux as _template_fit_deriv_rescale_only_flux,
)
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

### By CONVOLUTION

cdef inline void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

cdef inline void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

cdef inline void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_exact_all(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_all(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

### By INTERPOLATION

cdef inline void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

cdef inline void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

cdef inline void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_interp_all(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_all(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

### By RESCALING

cdef inline void _fit_deriv_rescale_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_fit_deriv_rescale_only_flux(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_rescale_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_rescale_only_flux(
        derivs, 
        flux, fwhm,
        cytemplate, n_scales,
    )