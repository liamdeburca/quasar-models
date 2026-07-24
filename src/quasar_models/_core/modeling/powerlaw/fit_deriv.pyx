from libc.math cimport (
    log as math_log,
    exp as math_exp,
)

###

cdef inline void _fit_deriv_only_flux(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil:
    cdef double log_x0 = math_log(x0)
    
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        derivs[0, i] += math_exp(alpha * (math_log(x[i]) - log_x0))

def fit_deriv_only_flux(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
):
    """
    Inplace derivative w.r.t. `flux`.
    """
    _fit_deriv_only_flux(derivs, x, flux, alpha, x0)

cdef inline void _fit_deriv_only_alpha(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil:
    cdef double log_x0 = math_log(x0)
    
    cdef double f
    cdef double log_ratio

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        log_ratio = math_log(x[i]) - log_x0
        f = flux * math_exp(alpha * log_ratio)
        derivs[1, i] += f * log_ratio

def fit_deriv_only_alpha(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
):
    """
    Inplace derivative w.r.t. `alpha`.
    """
    _fit_deriv_only_alpha(derivs, x, flux, alpha, x0)

cdef inline void _fit_deriv_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil:
    cdef double log_x0 = math_log(x0)

    cdef double log_ratio
    cdef double df_dflux

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        log_ratio = math_log(x[i]) - log_x0
        df_dflux = math_exp(alpha * log_ratio)

        derivs[0, i] += df_dflux
        derivs[1, i] += flux * df_dflux * log_ratio

def fit_deriv_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
):
    """
    Inplace derivative w.r.t. all parameters.
    """
    _fit_deriv_all(derivs, x, flux, alpha, x0)