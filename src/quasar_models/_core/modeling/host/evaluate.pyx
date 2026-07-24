from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact,
    _evaluate_interp as _template_evaluate_interp,
)

cdef inline _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _template_evaluate_exact(
        y,
        flux, fwhm, 
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

def evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _evaluate_exact(
        y,
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )

cdef inline _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
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
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    _evaluate_interp(
        y,
        flux, fwhm,
        template_fwhm, template_x, template_data,
        sigma_res, n_scales,
    )