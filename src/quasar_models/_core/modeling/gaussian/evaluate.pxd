cdef void _evaluate_v(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _prime_v(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _evaluate_x(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _prime_x(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil