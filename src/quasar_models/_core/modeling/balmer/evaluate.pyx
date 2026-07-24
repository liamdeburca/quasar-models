from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact, 
    _evaluate_interp as _template_evaluate_interp,
)

cdef inline void _evaluate_exact(
    double[::1] y,
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
    _template_evaluate_exact(
        y,
        flux, fwhm,
        continuum_fwhm, template_x, continuum_data,
        sigma_res,
        n_scales,
    )
    if ratio > 0.0:
        _template_evaluate_exact(
            y,
            ratio * flux, fwhm,
            series_fwhm, template_x, series_data,
            sigma_res,
            n_scales,
        )

def evaluate_exact(
    double[::1] y,
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
        y,
        flux, fwhm, ratio,
        template_x,
        continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, 
        n_scales,
    )

cdef inline void _evaluate_interp(
    double[::1] y,
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
    _template_evaluate_interp(
        y,
        flux, fwhm,
        continuum_fwhm, template_x, continuum_data,
        sigma_res,
        n_scales,
    )
    if ratio > 0.0:
        _template_evaluate_interp(
            y,
            ratio * flux, fwhm,
            series_fwhm, template_x, series_data,
            sigma_res,
            n_scales,
        )

def evaluate_interp(
    double[::1] y,
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
        y,
        flux, fwhm, ratio,
        template_x,
        continuum_fwhm, continuum_data,
        series_fwhm, series_data,
        sigma_res, 
        n_scales,
    )