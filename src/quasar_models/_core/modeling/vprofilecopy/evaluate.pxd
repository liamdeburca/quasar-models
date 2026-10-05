cdef void _evaluate_v(
    double[::1] y,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _evaluate_x(
    double[::1] y,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double dx,
) noexcept nogil