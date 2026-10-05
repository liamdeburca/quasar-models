from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact, 
    _evaluate_interp as _template_evaluate_interp,
    _evaluate_rescale as _template_evaluate_rescale,
)
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

cdef inline void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _template_evaluate_exact(
        y,
        flux, fwhm,
        continuum_cytemplate,
        n_scales,
    )
    if ratio > 0.0:
        _template_evaluate_exact(
            y,
            ratio * flux, fwhm,
            series_cytemplate,
            n_scales,
        )

def evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _evaluate_exact(
        y,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _template_evaluate_interp(
        y,
        flux, fwhm,
        continuum_cytemplate,
        n_scales,
    )
    if ratio > 0.0:
        _template_evaluate_interp(
            y,
            ratio * flux, fwhm,
            series_cytemplate,
            n_scales,
        )

def evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _evaluate_interp(
        y,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )

cdef inline void _evaluate_rescale(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _template_evaluate_rescale(
        y,
        flux, fwhm,
        continuum_cytemplate,
        n_scales,
    )
    if ratio > 0.0:
        _template_evaluate_rescale(
            y,
            ratio * flux, fwhm,
            series_cytemplate,
            n_scales,
        )

def evaluate_rescale(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
):
    _evaluate_rescale(
        y,
        flux, fwhm, ratio,
        continuum_cytemplate,
        series_cytemplate,
        n_scales,
    )