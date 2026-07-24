from libc.math cimport (
    hypot as math_hypot,
    exp as math_exp,
    sqrt as math_sqrt,
    log as math_log,
    pi as math_pi,
)

cdef double GAUSS_AMP = 1 / math_sqrt(2 * math_pi)
cdef double SIGMA_TO_FWHM = 2 * math_sqrt(2 * math_log(2))
cdef double FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM

###

cdef inline void _evaluate_v(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace evaluation of a Gaussian function.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma = 1 / (mean * math_hypot(sigma_v, sigma_res))
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double z
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        y[i] += norm * math_exp(-0.5 * z * z)

def evaluate_v(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace evaluation of a Gaussian function.
    """
    _evaluate_v(y, x, strength, fwhm_v, v_off, wave, sigma_res)

###

cdef inline void _prime_v(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace evaluation of the derivative of a Gaussian function.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma = 1 / (mean * math_hypot(sigma_v, sigma_res))
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double z
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        y[i] += norm * math_exp(-0.5 * z * z) * (-z) * inv_sigma

def prime_v(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace evaluation of a Gaussian first spatial derivative.
    """
    _prime_v(y, x, strength, fwhm_v, v_off, wave, sigma_res)