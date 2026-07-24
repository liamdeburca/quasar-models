from quasar_models._core.utils cimport (
    multiply_and_add_to,
)
from quasar_models._core.convolution.utils cimport (
    _convolve,
    _identify_closest_idx,
)

### By CONVOLUTION

cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    cdef Py_ssize_t i, n = template_x.shape[0]
    cdef double[::1] f = _convolve(
        template_data,
        template_fwhm,
        fwhm,
        sigma_res,
        n_scales,
    )
    multiply_and_add_to(y, f, flux)

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

### By INTERPOLATION

cdef void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    cdef double fwhm0, fwhm1, grad
    cdef const double[::1] f0, f1
    cdef Py_ssize_t i, n = template_x.shape[0]

    cdef int idx = _identify_closest_idx(template_fwhm, fwhm)
    if template_fwhm[idx] == fwhm:
        f0 = template_data[idx, :]
        multiply_and_add_to(y, f0, flux)
    else:
        if idx == template_fwhm.shape[0] - 1:
            idx -= 1

        fwhm0 = template_fwhm[idx]
        fwhm1 = template_fwhm[idx + 1]
        grad = (fwhm - fwhm0) / (fwhm1 - fwhm0)

        f0 = template_data[idx, :]
        f1 = template_data[idx + 1, :]

        multiply_and_add_to(y, f0, flux * (1.0 - grad))
        multiply_and_add_to(y, f1, flux * grad)

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