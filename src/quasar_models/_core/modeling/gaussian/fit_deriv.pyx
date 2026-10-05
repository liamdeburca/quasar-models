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
cdef double C_KMS = 299792.458  # Speed of light in km/s
cdef double INV_C_KMS = 1 / C_KMS

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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma = 1.0 / (mean * math_hypot(sigma_v_c, sigma_res))
    
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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma_tot = 1.0 / math_hypot(sigma_v_c, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    cdef double norm = strength * GAUSS_AMP * inv_sigma
    
    cdef double k = norm * fwhm_v_c * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2) * INV_C_KMS

    cdef double z_sq
    cdef double exp_term
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z_sq = math_pow((x[i] - mean) * inv_sigma, 2)
        exp_term = math_exp(-0.5 * z_sq)
        dy[i] += exp_term * (z_sq - 1) * k

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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma = 1.0 / (mean * math_hypot(sigma_v_c, sigma_res))
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double l = norm * INV_C_KMS / (1.0 + v_off_c)

    cdef double z
    cdef double exp_term

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        exp_term = math_exp(-0.5 * z * z)
        dy[i] += exp_term * (z * x[i] * inv_sigma - 1) * l

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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma_tot = 1.0 / math_hypot(sigma_v_c, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double k = fwhm_v_c * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2) * INV_C_KMS
    
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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma = 1.0 / (mean * math_hypot(sigma_v_c, sigma_res))
    
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm
    cdef double l = norm * INV_C_KMS / (1.0 + v_off_c)
    
    cdef double z
    cdef double exp_term

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        exp_term = math_exp(-0.5 * z * z)
        
        derivs[0, i] += dnorm * exp_term
        derivs[2, i] += exp_term * (z * x[i] * inv_sigma - 1) * l

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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma_tot = 1.0 / math_hypot(sigma_v_c, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    cdef double norm = strength * GAUSS_AMP * inv_sigma
    
    cdef double k = norm * fwhm_v_c * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2) * INV_C_KMS
    cdef double l = norm * INV_C_KMS / (1.0 + v_off_c)
    
    cdef double z
    cdef double z_sq
    cdef double exp_term
    
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)

        derivs[1, i] += exp_term * (z_sq - 1.0) * k
        derivs[2, i] += exp_term * (z * x[i] * inv_sigma - 1) * l

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
    cdef double fwhm_v_c = fwhm_v * INV_C_KMS
    cdef double v_off_c = v_off * INV_C_KMS

    cdef double sigma_v_c = fwhm_v_c * FWHM_TO_SIGMA
    cdef double mean = wave * (1.0 + v_off_c)
    cdef double inv_sigma_tot = 1.0 / math_hypot(sigma_v_c, sigma_res)
    cdef double inv_sigma = inv_sigma_tot / mean
    
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm
    cdef double k = norm * fwhm_v_c * math_pow(inv_sigma_tot * FWHM_TO_SIGMA, 2) * INV_C_KMS
    cdef double l = norm * INV_C_KMS / (1.0 + v_off_c)
    
    cdef double z
    cdef double z_sq
    cdef double exp_term

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        
        derivs[0, i] += dnorm * exp_term
        derivs[1, i] += exp_term * (z_sq - 1) * k
        derivs[2, i] += exp_term * (z * x[i] * inv_sigma - 1) * l

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

### Wavelength-based resolution: sigma_res -> dx

cdef inline void _fit_deriv_x_strength(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivative w.r.t. `strength`.
    """
    cdef double sigma_v_c = fwhm_v * FWHM_TO_SIGMA * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double inv_sigma = 1.0 / math_hypot(mean * sigma_v_c, dx)
    cdef double dnorm = GAUSS_AMP * inv_sigma

    cdef double z
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        dy[i] += dnorm * math_exp(-0.5 * z * z)

cdef inline void _fit_deriv_x_fwhm_v(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivative w.r.t. `fwhm_v`.
    """
    cdef double sigma_v_c = fwhm_v * FWHM_TO_SIGMA * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double sigma_x = mean * sigma_v_c
    cdef double sigma = math_hypot(sigma_x, dx)
    cdef double inv_sigma = 1.0 / sigma
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double k = sigma_x * mean * FWHM_TO_SIGMA * INV_C_KMS * inv_sigma**2
    
    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        f = norm * exp_term

        dy[i] += f * k * (z_sq - 1)

cdef inline void _fit_deriv_x_v_off(
    double[::1] dy,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivative w.r.t. `v_off`.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double sigma_v_c = sigma_v * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double sigma_x = mean * sigma_v_c
    cdef double inv_sigma = 1.0 / math_hypot(sigma_x, dx)
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double l = wave * INV_C_KMS * inv_sigma
    cdef double m = sigma_x * sigma_v * INV_C_KMS * inv_sigma

    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        f = norm * exp_term

        dy[i] += f * l * (z + m * (z_sq - 1))

### Special cases: only calculate a single derivative

cdef inline void _fit_deriv_x_only_strength(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    _fit_deriv_x_strength(derivs[0,:], x, strength, fwhm_v, v_off, wave, dx)

def fit_deriv_x_only_strength(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivative w.r.t. `strength`.
    """
    _fit_deriv_x_only_strength(derivs, x, strength, fwhm_v, v_off, wave, dx)

cdef inline void _fit_deriv_x_only_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    _fit_deriv_x_fwhm_v(derivs[1,:], x, strength, fwhm_v, v_off, wave, dx)

def fit_deriv_x_only_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivative w.r.t. `fwhm_v`.
    """
    _fit_deriv_x_only_fwhm_v(derivs, x, strength, fwhm_v, v_off, wave, dx)

cdef inline void _fit_deriv_x_only_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    _fit_deriv_x_v_off(derivs[2,:], x, strength, fwhm_v, v_off, wave, dx)

def fit_deriv_x_only_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivative w.r.t. `v_off`.
    """
    _fit_deriv_x_only_v_off(derivs, x, strength, fwhm_v, v_off, wave, dx)

### Special cases: calculate two derivatives in parallel

cdef inline void _fit_deriv_x_strength_and_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. `strength` and `fwhm_v` in parallel.
    """
    cdef double sigma_v_c = fwhm_v * FWHM_TO_SIGMA * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double sigma_x = mean * sigma_v_c
    cdef double sigma = math_hypot(sigma_x, dx)
    cdef double inv_sigma = 1.0 / sigma
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm

    cdef double k = sigma_x * mean * FWHM_TO_SIGMA * INV_C_KMS * inv_sigma**2

    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        f = norm * exp_term

        derivs[0,i] += dnorm * exp_term
        derivs[1,i] += f * k * (z_sq - 1)

def fit_deriv_x_strength_and_fwhm_v(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivatives w.r.t. `strength` and `fwhm_v` in parallel.
    """
    _fit_deriv_x_strength_and_fwhm_v(derivs, x, strength, fwhm_v, v_off, wave, dx)

cdef inline void _fit_deriv_x_strength_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. `strength` and `v_off` in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double sigma_v_c = sigma_v * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double sigma_x = mean * sigma_v_c
    cdef double sigma = math_hypot(sigma_x, dx)
    cdef double inv_sigma = 1.0 / sigma
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm

    cdef double l = wave * INV_C_KMS * inv_sigma
    cdef double m = sigma_x * sigma_v * INV_C_KMS * inv_sigma

    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        f = norm * exp_term

        derivs[0,i] += dnorm * exp_term
        derivs[2,i] += f * l * (z + m * (z_sq - 1))

def fit_deriv_x_strength_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivatives w.r.t. `strength` and `v_off` in parallel.
    """
    _fit_deriv_x_strength_and_v_off(derivs, x, strength, fwhm_v, v_off, wave, dx)

cdef inline void _fit_deriv_x_fwhm_v_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. `fwhm_v` and `v_off` in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double sigma_v_c = sigma_v * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double sigma_x = mean * sigma_v_c
    cdef double sigma = math_hypot(sigma_x, dx)
    cdef double inv_sigma = 1.0 / sigma
    cdef double norm = strength * GAUSS_AMP * inv_sigma

    cdef double k = sigma_x * mean * FWHM_TO_SIGMA * INV_C_KMS * inv_sigma**2
    cdef double l = wave * INV_C_KMS * inv_sigma
    cdef double m = sigma_x * sigma_v * INV_C_KMS * inv_sigma

    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        f = norm * exp_term

        derivs[1,i] += f * k * (z_sq - 1)
        derivs[2,i] += f * l * (z + m * (z_sq - 1))

def fit_deriv_x_fwhm_v_and_v_off(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivatives w.r.t. `fwhm_v` and `v_off` in parallel.
    """
    _fit_deriv_x_fwhm_v_and_v_off(derivs, x, strength, fwhm_v, v_off, wave, dx)

### Calculate all derivatives

cdef inline void _fit_deriv_x_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
) noexcept nogil:
    """
    Inplace derivatives w.r.t. all three parameters in parallel.
    """
    cdef double sigma_v = fwhm_v * FWHM_TO_SIGMA
    cdef double sigma_v_c = sigma_v * INV_C_KMS
    cdef double mean = wave * (1.0 + v_off * INV_C_KMS)
    cdef double sigma_x = mean * sigma_v_c
    cdef double sigma = math_hypot(sigma_x, dx)
    cdef double inv_sigma = 1.0 / sigma
    cdef double dnorm = GAUSS_AMP * inv_sigma
    cdef double norm = strength * dnorm

    cdef double k = sigma_x * mean * FWHM_TO_SIGMA * INV_C_KMS * inv_sigma**2
    cdef double l = wave * INV_C_KMS * inv_sigma
    cdef double m = sigma_x * sigma_v * INV_C_KMS * inv_sigma

    cdef double z, z_sq, exp_term, f
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        z = (x[i] - mean) * inv_sigma
        z_sq = z * z
        exp_term = math_exp(-0.5 * z_sq)
        f = norm * exp_term

        derivs[0,i] += dnorm * exp_term
        derivs[1,i] += f * k * (z_sq - 1)
        derivs[2,i] += f * l * (z + m * (z_sq - 1))

def fit_deriv_x_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double strength,
    const double fwhm_v,
    const double v_off,
    const double wave,
    const double dx,
):
    """
    Inplace derivatives w.r.t. all three parameters in parallel.
    """
    _fit_deriv_x_all(derivs, x, strength, fwhm_v, v_off, wave, dx)
