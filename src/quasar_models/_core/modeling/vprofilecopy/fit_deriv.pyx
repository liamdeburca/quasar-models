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
    _evaluate_x as _gaussian_evaluate_x,
)

cdef double GAUSS_AMP = 1 / math_sqrt(2 * math_pi)
cdef double SIGMA_TO_FWHM = 2 * math_sqrt(2 * math_log(2))
cdef double FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM
cdef double C_KMS = 299792.458  # Speed of light in km/s
cdef double INV_C_KMS = 1.0 / C_KMS

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
    cdef double strength, fwhm_v_c, v_off_c, sigma_v_c, mean
    cdef double inv_sigma_tot, inv_sigma, dnorm, norm, k
    cdef double xj, z, z_sq, exp_term 
    cdef double f, df_dstrength, df_dfwhm, df_dvoff
    cdef Py_ssize_t i, j, n = strengths.shape[0], m = x.shape[0]

    for i in range(n):
        strength = strengths[i]
        fwhm_v_c = fwhm_vs[i] * INV_C_KMS
        v_off_c = v_offs[i] * INV_C_KMS

        sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
        mean = wave * (1.0 + v_off_c)
        inv_sigma_tot = 1.0 / math_hypot(sigma_v_c, sigma_res)
        inv_sigma = inv_sigma_tot / mean

        dnorm = GAUSS_AMP * inv_sigma
        norm = strength * dnorm

        k = norm * fwhm_v_c * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2) * INV_C_KMS
        l = norm * INV_C_KMS / (1.0 + v_off_c)

        for j in range(m):
            xj = x[j]
            z = (xj - mean) * inv_sigma
            z_sq = z * z
            exp_term = math_exp(-0.5 * z_sq)

            f = norm * exp_term
            df_dstrength = dnorm * exp_term
            df_dfwhm_v = exp_term * (z_sq - 1) * k
            df_dv_off = exp_term * (z * xj * inv_sigma - 1) * l

            derivs[0, j] += f
            derivs[1+3*i, j] += strength_scale * df_dstrength
            derivs[2+3*i, j] += strength_scale * df_dfwhm_v
            derivs[3+3*i, j] += strength_scale * df_dv_off

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

###

cdef inline void _fit_deriv_x_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double dx,
) noexcept nogil:
    cdef double strength, fwhm_v, v_off
    cdef double sigma_v, sigma_v_c, mean, sigma_x, sigma, inv_sigma, dnorm, norm
    cdef double k1, k2, k3
    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, j, n = strengths.shape[0], m = x.shape[0]

    for i in range(n):
        strength = strengths[i]
        fwhm_v = fwhm_vs[i]
        v_off = v_offs[i]

        sigma_v = fwhm_v * FWHM_TO_SIGMA
        sigma_v_c = sigma_v * INV_C_KMS
        mean = wave * (1.0 + v_off * INV_C_KMS)
        sigma_x = mean * sigma_v_c
        sigma = math_hypot(sigma_x, dx)
        inv_sigma = 1.0 / sigma
        dnorm = GAUSS_AMP * inv_sigma
        norm = strength * dnorm

        k1 = sigma_x * mean * FWHM_TO_SIGMA * INV_C_KMS * inv_sigma**2
        k2 = wave * INV_C_KMS * inv_sigma
        k3 = sigma_x * sigma_v * INV_C_KMS * inv_sigma

        for j in range(m):
            z = (x[j] - mean) * inv_sigma
            z_sq = z * z
            exp_term = math_exp(-0.5 * z_sq)
            f = norm * exp_term

            derivs[0,j] += f
            derivs[1+3*i,j] += strength_scale * dnorm * exp_term
            derivs[2+3*i,j] += strength_scale * f * k1 * (z_sq - 1)
            derivs[3+3*i,j] += strength_scale * f * k2 * (z + k3 * (z_sq -1))

def fit_deriv_x_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double dx,
):
    _fit_deriv_x_all(
        derivs, x,
        strength_scale,
        strengths, fwhm_vs, v_offs,
        wave, dx,
    )

cdef inline void _fit_deriv_x_only_strength_scale(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double dx,
) noexcept nogil:
    cdef Py_ssize_t i, n = strengths.shape[0]

    for i in range(n):
        _gaussian_evaluate_x(
            derivs[0, :],
            x,
            strengths[i],
            fwhm_vs[i],
            v_offs[i],
            wave,
            dx,
        )

def fit_deriv_x_only_strength_scale(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength_scale,
    const double[::1] strengths,
    const double[::1] fwhm_vs,
    const double[::1] v_offs,
    const double wave,
    const double dx,
):
    _fit_deriv_x_only_strength_scale(
        derivs, x,
        strength_scale,
        strengths, fwhm_vs, v_offs,
        wave, dx,
    )