from libc.math cimport (
    fmin,
    fmax,
    log as math_log,
    exp as math_exp,
)
from quasar_models._core.utils cimport (
    dbl_add_inplace,
)

cdef double clip(double val, double min_val, double max_val) noexcept nogil:
    return fmin(fmax(val, min_val), max_val)

cdef void _evaluate(
    double[::1] y,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
) noexcept nogil:
    cdef double k, a, zi
    cdef Py_ssize_t i, n = x.shape[0]

    if left == right:
        dbl_add_inplace(y, left)
    else:
        k = 1.0 / (sigma_res * scale)
        a = math_log(split) * k

        for i in range(n):
            zi = clip(a - k * math_log(x[i]), -5.0, 5.0)
            y[i] += (right - left) / (1.0 + math_exp(zi)) + left

### Python API

def evaluate(
    double[::1] y,
    const double[::1] x,
    const double split,
    const double left,
    const double right,
    const double sigma_res,
    const double scale,
):
    _evaluate(y, x, split, left, right, sigma_res, scale)