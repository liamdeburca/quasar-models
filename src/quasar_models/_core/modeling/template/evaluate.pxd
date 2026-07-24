cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
)

cdef void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
)