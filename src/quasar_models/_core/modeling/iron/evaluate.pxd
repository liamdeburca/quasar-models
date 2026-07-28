from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

cdef void _evaluate_exact_no_split(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)

cdef void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    CyTemplate cytemplate,
    const double scale,
    const double n_scales,
)