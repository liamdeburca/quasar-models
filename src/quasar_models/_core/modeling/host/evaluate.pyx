from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact,
    _evaluate_interp as _template_evaluate_interp,
)
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

cdef inline void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_evaluate_exact(
        y,
        flux, fwhm, 
        cytemplate, n_scales,
    )

def evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _evaluate_exact(
        y,
        flux, fwhm,
        cytemplate, n_scales,
    )

cdef inline void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _template_evaluate_interp(
        y,
        flux, fwhm, 
        cytemplate, n_scales,
    )

def evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _evaluate_interp(
        y,
        flux, fwhm,
        cytemplate, n_scales,
    )