from libc.math cimport (
    fmin,
    fmax,
    log as math_log,
    exp as math_exp,
    fabs as math_fabs,
)

cdef inline double clip(
    double val, 
    double min_val, 
    double max_val,
) noexcept nogil:
    return fmin(fmax(val, min_val), max_val)

### Special cases: only one free parameter

cdef void _fit_deriv_only_split(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, exp_z, s, ds_dz
    cdef double _const = (right - left) * k / split

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        if math_fabs(zi) < 5.0:
            exp_z = math_exp(zi)
            s = 1.0 / (1.0 + exp_z)

            ds_dz = -s * s * exp_z
            derivs[0, i] = _const * ds_dz

def fit_deriv_only_split(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_only_split(derivs, x, split, left, right, sigma_res, scale)

cdef void _fit_deriv_only_left(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, s

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        s = 1.0 / (1.0 + math_exp(zi))
        derivs[1, i] = -s + 1.0

def fit_deriv_only_left(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_only_left(derivs, x, split, left, right, sigma_res, scale)

cdef void _fit_deriv_only_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, s

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        s = 1.0 / (1.0 + math_exp(zi))

        derivs[2, i] = s

def fit_deriv_only_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_only_right(derivs, x, split, left, right, sigma_res, scale)

### Special cases: only two free parameters

cdef void _fit_deriv_split_and_left(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, exp_z, s, ds_dz
    cdef double _const = (right - left) * k / split

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        exp_z = math_exp(zi)
        s = 1.0 / (1.0 + exp_z)

        if math_fabs(zi) < 5.0:
            ds_dz = -s * s * exp_z
            derivs[0, i] = _const * ds_dz

        derivs[1, i] = -s + 1.0

def fit_deriv_split_and_left(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_split_and_left(derivs, x, split, left, right, sigma_res, scale)

cdef void _fit_deriv_split_and_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, exp_z, s, ds_dz
    cdef double _const = (right - left) * k / split

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        exp_z = math_exp(zi)
        s = 1.0 / (1.0 + exp_z)

        if math_fabs(zi) < 5.0:
            ds_dz = -s * s * exp_z
            derivs[0, i] = _const * ds_dz

        derivs[2, i] = s

def fit_deriv_split_and_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_split_and_right(derivs, x, split, left, right, sigma_res, scale)

cdef void _fit_deriv_left_and_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, exp_z, s

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        exp_z = math_exp(zi)
        s = 1.0 / (1.0 + exp_z)

        derivs[1, i] = -s + 1.0
        derivs[2, i] = s

def fit_deriv_left_and_right(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_left_and_right(derivs, x, split, left, right, sigma_res, scale)

### General case: all three free parameters

cdef void _fit_deriv_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k = 1.0 / (sigma_res * scale)
    cdef double a = math_log(split) * k

    cdef double zi, exp_z, s, ds_dz
    cdef double _const = (right - left) * k / split

    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
        exp_z = math_exp(zi)
        s = 1.0 / (1.0 + exp_z)

        if math_fabs(zi) < 5.0:
            ds_dz = -s * s * exp_z
            derivs[0, i] = _const * ds_dz

        derivs[1, i] = -s + 1.0
        derivs[2, i] = s

def fit_deriv_all(
    double[:,::1] derivs,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _fit_deriv_all(derivs, x, split, left, right, sigma_res, scale)