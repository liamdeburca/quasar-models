cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
)

cdef void _evaluate_interp(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double ratio,
    const double[::1] template_x,
    const double[::1] continuum_fwhm,
    const double[:,::1] continuum_data,
    const double[::1] series_fwhm,
    const double[:,::1] series_data,
    const double sigma_res,
    const double n_scales,
)