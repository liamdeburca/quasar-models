from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
)

cdef void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    CyTemplate continuum_cytemplate,
    CyTemplate series_cytemplate,
    const double n_scales,
)