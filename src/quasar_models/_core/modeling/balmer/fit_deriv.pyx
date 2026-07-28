from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact,
    _evaluate_interp as _template_evaluate_interp,
)
from quasar_models._core.modeling.template.fit_deriv cimport (
    _fit_deriv_exact_only_fwhm as _template_fit_deriv_exact_only_fwhm,
    _fit_deriv_interp_only_fwhm as _template_fit_deriv_interp_only_fwhm,
)
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate
from quasar_models._core.modeling.balmer.evaluate cimport (
    _evaluate_exact,
    _evaluate_interp,
)

### By CONVOLUTION

# Special cases: only one free parameter

cdef inline void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _evaluate_exact(
        derivs[0, :],
        1.0, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :], 
        flux, fwhm,
        continuum_cytemplate, n_scales
    )
    if ratio != 0.0:
        _template_fit_deriv_exact_only_fwhm(
            derivs[0:2, :], 
            ratio * flux, fwhm,
            series_cytemplate, n_scales
        )

cdef inline void _fit_deriv_exact_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    if flux != 0.0:
        _template_evaluate_exact(
            derivs[2, :],
            flux, fwhm,
            series_cytemplate,
            n_scales,
        )

# Special cases: two free parameters

cdef inline void _fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_exact_only_ratio(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_exact_only_ratio(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

# General case

cdef inline void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_exact_only_ratio(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

### By INTERPOLATION

# Special cases: only one free parameter

cdef inline void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _evaluate_interp(
        derivs[0, :],
        1.0, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _template_fit_deriv_interp_only_fwhm(
        derivs[0:2, :], 
        flux, fwhm,
        continuum_cytemplate, n_scales
    )
    if ratio != 0.0:
        _template_fit_deriv_interp_only_fwhm(
            derivs[0:2, :], 
            ratio * flux, fwhm,
            series_cytemplate, n_scales
        )

cdef inline void _fit_deriv_interp_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    if flux != 0.0:
        _template_evaluate_interp(
            derivs[2, :],
            flux, fwhm,
            series_cytemplate,
            n_scales,
        )

# Special cases: two free parameters

cdef inline void _fit_deriv_interp_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _fit_deriv_interp_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_interp_only_ratio(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _fit_deriv_interp_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_interp_only_ratio(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

# General case

cdef inline void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
    _fit_deriv_interp_only_ratio(
        derivs,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

# ======== PUBLIC PYTHON INTERFACE ========

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_exact_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_ratio(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_exact_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_exact_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_all(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_ratio(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_flux_and_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_flux_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_fwhm_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_all(
        derivs, 
        flux, fwhm, ratio, 
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )
