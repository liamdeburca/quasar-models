cdef void _evaluate(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double fwhm,
    const double sigma_res,
    const double[::1] waves,
    const double[::1] weights,
)