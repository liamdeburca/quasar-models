cdef void _fit_deriv_only_split(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil

cdef void _fit_deriv_only_left(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil

cdef void _fit_deriv_only_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil

cdef void _fit_deriv_split_and_left(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil

cdef void _fit_deriv_split_and_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil

cdef void _fit_deriv_left_and_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil

cdef void _fit_deriv_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil