cdef void _fit_deriv_v_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_only_strength_scale(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
) noexcept nogil
