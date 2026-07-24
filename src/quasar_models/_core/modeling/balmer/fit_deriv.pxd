### By CONVOLUTION

cdef void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
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

cdef void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
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

cdef void _fit_deriv_exact_only_ratio(
    double[:,::1] derivs,
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

cdef void _fit_deriv_exact_flux_and_fwhm(
    double[:,::1] derivs,
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

cdef void _fit_deriv_exact_flux_and_ratio(
    double[:,::1] derivs,
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

cdef void _fit_deriv_exact_fwhm_and_ratio(
    double[:,::1] derivs,
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

cdef void _fit_deriv_exact_all(
    double[:,::1] derivs,
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

### By INTERPOLATION

### By CONVOLUTION

cdef void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
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

cdef void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
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

cdef void _fit_deriv_interp_only_ratio(
    double[:,::1] derivs,
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

cdef void _fit_deriv_interp_flux_and_fwhm(
    double[:,::1] derivs,
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

cdef void _fit_deriv_interp_flux_and_ratio(
    double[:,::1] derivs,
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

cdef void _fit_deriv_interp_fwhm_and_ratio(
    double[:,::1] derivs,
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

cdef void _fit_deriv_interp_all(
    double[:,::1] derivs,
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
