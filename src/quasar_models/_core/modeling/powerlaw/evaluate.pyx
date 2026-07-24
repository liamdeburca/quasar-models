from libc.math cimport (
    log as math_log,
    exp as math_exp,
)

###

cdef inline void _evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil:
    cdef double log_x0 = math_log(x0)
    cdef Py_ssize_t i, n = x.shape[0]

    for i in range(n):
        y[i] += flux * math_exp(alpha * (math_log(x[i]) - log_x0))

def evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
):
    _evaluate(y, x, flux, alpha, x0)

cdef inline void _inverse(
    double[::1] x,
    const double[::1] y,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil:
    cdef double log_flux = math_log(flux)
    cdef double inv_alpha = 1.0 / alpha

    cdef Py_ssize_t i, n = y.shape[0]
    for i in range(n):
        x[i] += x0 * math_exp(inv_alpha * (math_log(y[i]) - log_flux))

def inverse(
    double[::1] x,
    const double[::1] y,
    const double flux,
    const double alpha,
    const double x0,
):
    _inverse(x, y, flux, alpha, x0)