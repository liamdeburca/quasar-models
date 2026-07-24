cdef void _attenuation(
    double[::1] y,
    const double[::1] x,
    const double tau,
    const double scale,
    const double edge,
) noexcept nogil

cdef void _continuum(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double temp,
    const double edge,
    const double boltz,
) noexcept nogil

cdef void _evaluate(
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
) noexcept nogil