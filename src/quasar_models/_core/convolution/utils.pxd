cdef double _scale(
    const double fwhm,
    const double sigma_res,
)

cdef object _pixels(
    const double scale,
    const double n_scales,
)

cdef object _kernel(
    const double fwhm,
    const double sigma_res,
    const double n_scales,
)

cdef object _kernel_deriv(
    const double fwhm,
    const double sigma_res,
    const double n_scales,
)

cdef object _convolve_signal(
    const double[::1] signal,
    const double[::1] kernel,
)

cdef object _convolve_signal2d(
    const double[:,::1] signal,
    const double[::1] kernel,
)

cdef int _identify_closest_idx(
    const double[::1] fwhm,
    const double fwhm_final,
)

cdef int _identify_closest_idx_for_deriv(
    const double[::1] fwhm,
    const double fwhm_final,
)

cdef object _convolve(
    const double[:,::1] data,
    const double[::1] fwhm,
    const double fwhm_final,
    const double sigma_res,
    const double n_scales,
)

cdef object _convolve_deriv(
    const double[:,::1] data,
    const double[::1] fwhm,
    const double fwhm_final,
    const double sigma_res,
    const double n_scales,
)