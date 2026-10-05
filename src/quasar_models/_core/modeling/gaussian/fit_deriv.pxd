cdef void _fit_deriv_v_strength(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_fwhm_v(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_v_off(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_only_strength(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_only_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_only_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_strength_and_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_strength_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_fwhm_v_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

cdef void _fit_deriv_v_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil

###

cdef void _fit_deriv_x_strength(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_fwhm_v(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_v_off(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_only_strength(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_only_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_only_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_strength_and_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_strength_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_fwhm_v_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil

cdef void _fit_deriv_x_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil