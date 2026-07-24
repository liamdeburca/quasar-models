from quasar_models._core.modeling.template.fit_deriv cimport (
    _fit_deriv_exact_only_flux as _template_fit_deriv_exact_only_flux,
    _fit_deriv_exact_only_fwhm as _template_fit_deriv_exact_only_fwhm,
    _fit_deriv_exact_all as _template_fit_deriv_exact_all,

    _fit_deriv_interp_only_flux as _template_fit_deriv_interp_only_flux,
    _fit_deriv_interp_only_fwhm as _template_fit_deriv_interp_only_fwhm,
    _fit_deriv_interp_all as _template_fit_deriv_interp_all,
)

### By CONVOLUTION

cdef inline _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data, 
        sigma_res, n_scales,
    )

cdef inline _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

cdef inline _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_exact_all(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_exact_all(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

### By INTERPOLATION

cdef inline _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

cdef inline _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

cdef inline _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_fit_deriv_interp_all(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _fit_deriv_interp_all(
        derivs, 
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )