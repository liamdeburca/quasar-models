cdef class CyTemplate:
    cdef readonly double[::1] fwhm
    cdef readonly double[::1] x
    cdef readonly double[:,::1] data
    cdef readonly double sigma_res

    cdef CyTemplate split(
        self,
        const double split,
        const double left,
        const double right,
        const double scale,
    )

    ### Split derivatives

    cdef void _add_split_fit_deriv(
        self,
        double[:,::1] derivs,
        double[:,::1] split_derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
        const int[:] indices,
    )

    cdef void add_split_fit_deriv_only_split(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )

    cdef void add_split_fit_deriv_only_left(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )

    cdef void add_split_fit_deriv_only_right(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )

    cdef void add_split_fit_deriv_split_and_left(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )

    cdef void add_split_fit_deriv_split_and_right(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )

    cdef void add_split_fit_deriv_left_and_right(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )

    cdef void add_split_fit_deriv_all(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    )