from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact,
    _evaluate_interp as _template_evaluate_interp,
)
from quasar_models._core.modeling.template.fit_deriv cimport (
    _fit_deriv_exact_only_fwhm as _template_fit_deriv_exact_only_fwhm,
    _fit_deriv_interp_only_fwhm as _template_fit_deriv_interp_only_fwhm,
)
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
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _evaluate_exact(
        derivs[0, :],
        1.0, fwhm, ratio,
        template_x,
        continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res,
        n_scales,
    )

cdef inline void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_exact_only_fwhm(
        derivs[0:2, :], 
        flux, fwhm,
        continuum_fwhm, template_x, continuum_data,
        sigma_res, n_scales
    )
    if ratio != 0.0:
        _template_fit_deriv_exact_only_fwhm(
            derivs[0:2, :], 
            ratio * flux, fwhm,
            series_fwhm, template_x, series_data,
            sigma_res, n_scales
        )

cdef inline void _fit_deriv_exact_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    if flux != 0.0:
        _template_evaluate_exact(
            derivs[2, :],
            flux, fwhm,
            series_fwhm, template_x, series_data,
            sigma_res, n_scales,
        )

# Special cases: two free parameters

cdef inline void _fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

cdef inline void _fit_deriv_exact_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_exact_only_ratio(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

cdef inline void _fit_deriv_exact_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_exact_only_ratio(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

# General case

cdef inline void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_exact_only_ratio(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

### By INTERPOLATION

# Special cases: only one free parameter

cdef inline void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _evaluate_interp(
        derivs[0, :],
        1.0, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res,
        n_scales,
    )

cdef inline void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_interp_only_fwhm(
        derivs[0:2, :], 
        flux, fwhm,
        continuum_fwhm, template_x, continuum_data,
        sigma_res, n_scales
    )
    if ratio != 0.0:
        _template_fit_deriv_interp_only_fwhm(
            derivs[0:2, :], 
            ratio * flux, fwhm,
            series_fwhm, template_x, series_data,
            sigma_res, n_scales
        )

cdef inline void _fit_deriv_interp_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    if flux != 0.0:
        _template_evaluate_interp(
            derivs[2, :],
            flux, fwhm,
            series_fwhm, template_x, series_data,
            sigma_res,
            n_scales,
        )

# Special cases: two free parameters

cdef inline void _fit_deriv_interp_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

cdef inline void _fit_deriv_interp_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_interp_only_ratio(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

cdef inline void _fit_deriv_interp_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_interp_only_ratio(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )

# General case

cdef inline void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
    _fit_deriv_interp_only_ratio(
        derivs,
        flux, fwhm, ratio,
        template_x, continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, n_scales,
    )
# ======== PUBLIC PYTHON INTERFACE ========

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_exact_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_ratio(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_exact_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_flux_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_exact_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_fwhm_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_all(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_only_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_ratio(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_flux_and_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_flux_and_fwhm(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_flux_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_flux_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_fwhm_and_ratio(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_fwhm_and_ratio(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_all(
        derivs, 
        flux, fwhm, ratio, 
        template_x, 
        continuum_fwhm, continuum_data, 
        series_fwhm, series_data, 
        sigma_res, n_scales,
    )
