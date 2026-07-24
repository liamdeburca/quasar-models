from libc.math cimport (
    hypot as math_hypot,
    exp as math_exp,
    sqrt as math_sqrt,
    log as math_log,
    pi as math_pi,
    pow as math_pow,
)

cdef double GAUSS_AMP = 1 / math_sqrt(2 * math_pi)
cdef double SIGMA_TO_FWHM = 2 * math_sqrt(2 * math_log(2))
cdef double FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM

cdef inline void _fit_deriv_v_strength(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivative w.r.t. `strength`.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma = 1 / (mean * math_hypot(sigma_v, sigma_res))
    
    cdef double dnorm = GAUSS_AMP * inv_sigma

    cdef double z
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        dy[i] += dnorm * math_exp(-0.5 * z * z)

cdef inline void _fit_deriv_v_fwhm_v(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivative w.r.t. `fwhm_v`.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma_tot = 1 / math_hypot(sigma_v, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    cdef double norm = strength * GAUSS_AMP * inv_sigma
    
    cdef double k = fwhm_v * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2)

    cdef double z_sq
    cdef double f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z_sq = math_pow((x[i] - mean) * inv_sigma, 2)
        f = norm * math_exp(-0.5 * z_sq)
        dy[i] += f * (z_sq - 1) * k

cdef inline void _fit_deriv_v_v_off(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivative w.r.t. `v_off`.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma = 1 / (mean * math_hypot(sigma_v, sigma_res))
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double z
    cdef double f

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        f = norm * math_exp(-0.5 * z * z)
        dy[i] += f * (z * x[i] * inv_sigma - 1) / (1 + v_off)

### Special cases: only calculate a single derivative

cdef inline void _fit_deriv_v_only_strength(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    _fit_deriv_v_strength(derivs[0,:], x, strength, fwhm_v, v_off, wave, sigma_res)

def fit_deriv_v_only_strength(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivative w.r.t. `strength`.
    """
    _fit_deriv_v_only_strength(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

cdef inline void _fit_deriv_v_only_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    _fit_deriv_v_fwhm_v(derivs[1,:], x, strength, fwhm_v, v_off, wave, sigma_res)

def fit_deriv_v_only_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivative w.r.t. `fwhm_v`.
    """
    _fit_deriv_v_only_fwhm_v(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

cdef inline void _fit_deriv_v_only_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    _fit_deriv_v_v_off(derivs[2,:], x, strength, fwhm_v, v_off, wave, sigma_res)

def fit_deriv_v_only_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivative w.r.t. `v_off`.
    """
    _fit_deriv_v_only_v_off(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

### Special cases: calculate two derivatives in parallel

cdef inline void _fit_deriv_v_strength_and_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. `strength` and `fwhm_v` in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma_tot = 1 / math_hypot(sigma_v, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double k = fwhm_v * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2)
    
    cdef double z_sq
    cdef double df_dstrength

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z_sq = math_pow((x[i] - mean) * inv_sigma, 2)
        df_dstrength = dnorm * math_exp(-0.5 * z_sq)

        derivs[0, i] += df_dstrength
        derivs[1, i] += strength * df_dstrength * (z_sq - 1) * k

def fit_deriv_v_strength_and_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivatives w.r.t. `strength` and `fwhm_v` in parallel.
    """
    _fit_deriv_v_strength_and_fwhm_v(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

cdef inline void _fit_deriv_v_strength_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. `strength` and `v_off` in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma = 1 / (mean * math_hypot(sigma_v, sigma_res))
    
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm
    
    cdef double z
    cdef double exp_term

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        exp_term = math_exp(-0.5 * z * z)
        
        derivs[0, i] += dnorm * exp_term
        derivs[2, i] += norm * exp_term * (z * x[i] * inv_sigma - 1) / (1 + v_off)

def fit_deriv_v_strength_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivatives w.r.t. `strength` and `v_off` in parallel.
    """
    _fit_deriv_v_strength_and_v_off(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

cdef inline void _fit_deriv_v_fwhm_v_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. `fwhm_v` and `v_off` in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma_tot = 1 / math_hypot(sigma_v, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    cdef double norm = strength * GAUSS_AMP * inv_sigma
    
    cdef double k = fwhm_v * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2)
    
    cdef double z
    cdef double z_sq
    cdef double exp_term
    
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)

        derivs[1, i] += norm * exp_term * (z_sq - 1) * k
        derivs[2, i] += norm * exp_term * (z * x[i] * inv_sigma - 1) / (1 + v_off)

def fit_deriv_v_fwhm_v_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivatives w.r.t. `fwhm_v` and `v_off` in parallel.
    """
    _fit_deriv_v_fwhm_v_and_v_off(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)

### Calculate all derivatives

cdef inline void _fit_deriv_v_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. all three parameters in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double mean = wave * (1 + v_off)
    cdef double inv_sigma_tot = 1 / math_hypot(sigma_v, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm
    cdef double k = fwhm_v * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2)
    
    cdef double z
    cdef double z_sq
    cdef double exp_term

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        
        derivs[0, i] += dnorm * exp_term
        derivs[1, i] += norm * exp_term * (z_sq - 1) * k
        derivs[2, i] += norm * exp_term * (z * x[i] * inv_sigma - 1) / (1 + v_off)

def fit_deriv_v_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double sigma_res,
):
    """
    Inplace derivatives w.r.t. all three parameters in parallel.
    """
    _fit_deriv_v_all(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)