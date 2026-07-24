ctypedef inline (*_evaluate_func)(
    double[::1] y,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
)

ctypedef inline (*_fit_deriv_func)(
    double[:,::1] derivs,
    const double[::1] x,
    const double flux,
    const double alpha,
    const double x0,
)

cdef class PowerLawModel:
    cdef const double x0
    cdef const _evaluate_func evaluate_func
    cdef const _fit_deriv_func fit_deriv_func

    def __init__(
        self,
        object python_powerlaw_model,
    ):
        self.x0 = python_powerlaw_model.x0
        self.evaluate_func = python_powerlaw_model.evaluate_func
        self.fit_deriv_func = python_powerlaw_model.fit_deriv_func

    cdef inline evaluate(
        double[::1] y,
        const double[::1] x,
        const double flux,
        const double alpha,
    ):
        self.evaluate_func(
            y, x,
            flux, alpha,
            self.x0,
        )

    cdef inline fit_deriv(
        double[:,::1] derivs,
        const double[::1] x,
        const double flux,
        const double alpha,
    ):
        self.fit_deriv_func(
            derivs, x,
            flux,  alpha,
            self.x0,
        )