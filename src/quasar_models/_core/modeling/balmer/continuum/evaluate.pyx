from libc.math cimport (
    expm1 as math_expm1,
    pow as math_pow,
)
from numpy import (
    array as np_array,
    float64 as np_float64,
    copy as np_copy,
)

from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact as _template_evaluate_exact,
)

cdef inline void _attenuation(
    double[::1] y,
    const double[::1] x,
    const double tau,
    const double scale,
    const double edge,
) noexcept nogil:
    cdef double k = -tau * math_pow(edge, -scale)
    cdef Py_ssize_t i, n = x.shape[0]

    if tau != 0.0:
        for i in range(n):
            if x[i] <= edge:
                y[i] -= math_expm1(k * math_pow(x[i], scale))

def attenuation(
    double[::1] y,
    const double[::1] x,
    const double tau,
    const double scale,
    const double edge,
):
    _attenuation(y, x, tau, scale, edge)

cdef inline void _continuum(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double temp,
    const double edge,
    const double boltz,
) noexcept nogil:
    cdef double xi, k = boltz / temp
    cdef Py_ssize_t i, n = x.shape[0]

    if flux != 0.0:
        for i in range(n):
            xi = x[i]
            if xi <= edge:
                y[i] += flux / (math_pow(xi, 5) * math_expm1(k / xi))

def continuum(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double temp,
    const double edge,
    const double boltz,
):
    _continuum(y, x, flux, temp, edge, boltz)

cdef inline void _evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double fwhm,
    const double temp,
    const double tau,
    const double scale,
    const double edge,
    const double boltz,
    const double sigma_res,
    const double n_scales,
) noexcept nogil:
    cdef double xi, ai, k = -tau * math_pow(edge, -scale), l = boltz / temp
    cdef const double[::1] template_fwhm
    cdef const double[:,::1] template_data

    cdef Py_ssize_t i, n = x.shape[0]

    if flux != 0.0:
        for i in range(n):
            xi = x[i]
            if xi <= edge:
                ai = -math_expm1(k * math_pow(xi, scale))
                y[i] += ai * flux / (math_pow(xi, 5) * math_expm1(l / xi))

    if fwhm > 0.0:
        with gil:
            template_fwhm = np_array([0.0], dtype=np_float64)
            template_data = np_copy(y)[None, :]

            _template_evaluate_exact(
                y, 
                1.0, fwhm,
                template_fwhm, x, template_data,
                sigma_res, n_scales,
            )

def evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double fwhm,
    const double temp,
    const double tau,
    const double scale,
    const double edge,
    const double boltz,
    const double sigma_res,
    const double n_scales,
):
    _evaluate(y, x, flux, fwhm, temp, tau, scale, edge, boltz, sigma_res, n_scales)