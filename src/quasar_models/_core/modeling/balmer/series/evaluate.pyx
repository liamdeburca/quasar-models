from libc.math cimport (
    log as math_log,
    sqrt as math_sqrt,
    pi as math_pi,
    exp as math_exp,
)

cdef double GAUSS_AMP = 1 / math_sqrt(2 * math_pi)
cdef double SIGMA_TO_FWHM = 2 * math_sqrt(2 * math_log(2))
cdef double FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM

cdef inline void _evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double fwhm,
    const double sigma_res,
    const double[::1] waves,
    const double[::1] weights,
):
    cdef double denom = math_log(1.0 + sigma_res)
    cdef double sigma_pix = FWHM_TO_SIGMA * fwhm / sigma_res
    cdef double inv_sigma_pix = 1.0 / sigma_pix
    cdef double amp = GAUSS_AMP * inv_sigma_pix

    cdef double _mu, weight, _xj, _zj

    cdef Py_ssize_t i, n = waves.shape[0]
    cdef Py_ssize_t j, m = x.shape[0]

    for i in range(n):
        _mu = math_log(waves[i]) / denom
        weight = weights[i]

        for j in range(m):
            _xj = math_log(x[j]) / denom
            _zj = (_xj - _mu) * inv_sigma_pix

            y[j] += flux * amp * weight * math_exp(-0.5 * _zj * _zj)

def evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double fwhm,
    const double sigma_res,
    const double[::1] waves,
    const double[::1] weights,
):
    _evaluate(y, x, flux, fwhm, sigma_res, waves, weights)