cdef void _fit_deriv_only_flux(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil

cdef void _fit_deriv_only_alpha(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil

cdef void _fit_deriv_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
) noexcept nogil