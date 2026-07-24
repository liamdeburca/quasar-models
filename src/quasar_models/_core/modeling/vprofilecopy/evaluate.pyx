from quasar_models._core.modeling.gaussian.evaluate cimport (
    _evaluate_v as _gaussian_evaluate_v,
)

cdef inline void _evaluate_v(
    double[::1] y,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    cdef Py_ssize_t i, n = strengths.shape[0]
    for i in range(n):
        _gaussian_evaluate_v(
            y, x,
            strengths[i] * strength_scale,
            fwhm_vs[i],
            v_offs[i],
            wave,
            sigma_res,
        )

def evaluate_v(
    double [::1] y,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
):
    _evaluate_v(
        y, x,
        strength_scale,
        strengths, fwhm_vs, v_offs,
        wave, sigma_res,
    )