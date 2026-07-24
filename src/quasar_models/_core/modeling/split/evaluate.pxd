cdef void _evaluate(
    double[::1] y,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil