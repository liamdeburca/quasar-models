ctypedef inline (*_evaluate_func)(
    double[::1] y,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
)

ctypedef inline (*_fit_deriv_func)(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
)

cdef class GaussianModel:
    cdef const double wave
    cdef const double sigma_res
    cdef const _evaluate_func evaluate_func
    cdef const _fit_deriv_func fit_deriv_func

    def __init__(
        self,
        object python_gaussian_model,
    ):
        self.wave = python_gaussian_model.wave
        self.sigma_res = python_gaussian_model.sigma_res
        self.evaluate_func = python_gaussian_model.evaluate_func
        self.fit_deriv_func = python_gaussian_model.fit_deriv_func

    cdef inline evaluate(
        self,
        double[::1] y,
        const double[::1] x,
        const double strength,
        const double fwhm_v,
        const double v_off,
    ):
        self.evaluate_func(
            y, x,
            strength, fwhm_v, v_off,
            self.wave, self.sigma_res,
        )

    cdef inline fit_deriv(
        self,
        double[:,::1] derivs,
        const double[::1] x,
        const double strength,
        const double fwhm_v,
        const double v_off,
    ):
        self.fit_deriv_func(
            derivs, x,
            strength, fwhm_v, v_off, 
            self.wave, self.sigma_res,
        )
