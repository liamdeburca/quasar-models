from quasar_models._core.utils cimport (
    multiply_and_add_to,
)
from quasar_models._core.convolution.utils cimport (
    _convolve,
    _identify_closest_idx,
)
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

### By CONVOLUTION

cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    cdef Py_ssize_t i, n = len(cytemplate.x)
    cdef double[::1] f = _convolve(
        cytemplate.data,
        cytemplate.fwhm,
        fwhm,
        cytemplate.sigma_res,
        n_scales,
    )
    multiply_and_add_to(y, f, flux)

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

### By INTERPOLATION

cdef void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    cdef double fwhm0, fwhm1, grad
    cdef const double[::1] f0, f1
    cdef Py_ssize_t i, n = len(cytemplate.x)

    cdef int idx = _identify_closest_idx(cytemplate.fwhm, fwhm)
    if cytemplate.fwhm[idx] == fwhm:
        f0 = cytemplate.data[idx, :]
        multiply_and_add_to(y, f0, flux)
    else:
        if idx == len(cytemplate.fwhm) - 1:
            idx -= 1

        fwhm0 = cytemplate.fwhm[idx]
        fwhm1 = cytemplate.fwhm[idx + 1]
        grad = (fwhm - fwhm0) / (fwhm1 - fwhm0)

        f0 = cytemplate.data[idx, :]
        f1 = cytemplate.data[idx + 1, :]

        multiply_and_add_to(y, f0, flux * (1.0 - grad))
        multiply_and_add_to(y, f1, flux * grad)

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