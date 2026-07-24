from libc.math cimport (
    hypot as math_hypot,
    exp as math_exp,
    sqrt as math_sqrt,
    pow as math_pow,
    log as math_log,
    pi as math_pi,
)
from quasar_models._core.modeling.gaussian.evaluate cimport (
    _evaluate_v as _gaussian_evaluate_v,
)

cdef double GAUSS_AMP = 1 / math_sqrt(2 * math_pi)
cdef double SIGMA_TO_FWHM = 2 * math_sqrt(2 * math_log(2))
cdef double FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM

cdef inline void _fit_deriv_v_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    cdef double strength, fwhm_v, v_off, sigma_v, mean, 
    cdef double inv_sigma, inv_sigma_tot, dnorm, norm, k
    cdef double xj, z, z_sq, exp_term, 
    cdef double f, df_dstrength, df_dfwhm, df_dvoff
    cdef Py_ssize_t i, j, n = strengths.shape[0], m = x.shape[0]

    for i in range(n):
        strength = strengths[i]
        fwhm_v = fwhm_vs[i]
        v_off = v_offs[i]

        sigma_v = fwhm_v * FWHM_TO_SIGMA
        mean = wave * (1 + v_off)
        inv_sigma_tot = 1 / math_hypot(sigma_v, sigma_res)
        inv_sigma = inv_sigma_tot / mean
        dnorm = GAUSS_AMP * inv_sigma
        norm = strength * dnorm

        k = fwhm_v * math_pow(inv_sigma_tot / SIGMA_TO_FWHM, 2)

        for j in range(m):
            xj = x[j]
            
            z = (xj - mean) * inv_sigma
            z_sq = z * z
            exp_term = math_exp(-0.5 * z_sq)

            f = norm * exp_term
            df_dstrength = dnorm * exp_term
            df_dfwhm = strength * df_dstrength * (z_sq - 1) * k
            df_dvoff = f * (z * xj * inv_sigma - 1) / (1 + v_off)

            derivs[0, j] += f
            derivs[1+3*i, j] += strength_scale * df_dstrength
            derivs[2+3*i, j] += strength_scale * df_dfwhm
            derivs[3+3*i, j] += strength_scale * df_dvoff

def fit_deriv_v_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
):
    _fit_deriv_v_all(
        derivs, x,
        strength_scale,
        strengths, fwhm_vs, v_offs,
        wave, sigma_res,
    )

cdef inline void _fit_deriv_v_only_strength_scale(
    double[:,::1] derivs,
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
            derivs[0, :],
            x,
            strengths[i],
            fwhm_vs[i],
            v_offs[i],
            wave,
            sigma_res,
        )

def fit_deriv_v_only_strength_scale(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double sigma_res,
):
    _fit_deriv_v_only_strength_scale(
        derivs, x,
        strength_scale,
        strengths, fwhm_vs, v_offs,
        wave, sigma_res,
    )
