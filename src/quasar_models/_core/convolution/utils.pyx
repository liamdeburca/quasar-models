from libc.math cimport (
    log as math_log,
    exp as math_exp,
    sqrt as math_sqrt,
    pi as math_pi,
)
from numpy import (
    arange as np_arange,
    ones as np_ones,
    zeros as np_zeros,
    full as np_full,
    float64 as np_float64,
    pad as np_pad,
    searchsorted as np_searchsorted,
    copy as np_copy,
)
from scipy.signal import fftconvolve

from quasar_models._core.utils cimport (
    arr_add_inplace,
    dbl_multiply_inplace,
    dbl_divide_inplace,
)

cdef double GAUSS_AMP = 1 / math_sqrt(2 * math_pi)
cdef double SIGMA_TO_FWHM = 2 * math_sqrt(2 * math_log(2))
cdef double FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM

cdef double _scale(
    const double fwhm,
    const double sigma_res,
):
    return fwhm * FWHM_TO_SIGMA / sigma_res

def scale(
    const double fwhm,
    const double sigma_res,
):
    return _scale(fwhm, sigma_res)

cdef object _pixels(
    const double scale,
    const double n_scales,
):
    cdef int l = int(n_scales * scale)
    return np_arange(-l, l + 1, dtype=np_float64)

def pixels(
    const double scale,
    const double n_scales,
):
    return _pixels(scale, n_scales)

cdef object _kernel(
    const double fwhm,
    const double sigma_res,
    const double n_scales,
):
    if fwhm == 0:
        return np_ones(1, dtype=np_float64)

    cdef double s = _scale(fwhm, sigma_res)
    
    cdef object z = _pixels(s, n_scales)
    dbl_divide_inplace(z, s)
    
    cdef object k = np_zeros(z.shape[0], dtype=np_float64)
    cdef double k_sum = 0.0

    cdef Py_ssize_t i, n = z.shape[0]
    for i in range(n):
        k[i] = math_exp(-0.5 * z[i] * z[i])
        k_sum += k[i]

    dbl_divide_inplace(k, k_sum)

    return k

def kernel(
    const double fwhm,
    const double sigma_res,
    const double n_scales,
):
    return _kernel(fwhm, sigma_res, n_scales)

cdef object _kernel_deriv(
    const double fwhm,
    const double sigma_res,
    const double n_scales,
):
    if fwhm == 0:
        return np_full(1, -FWHM_TO_SIGMA, dtype=np_float64)

    cdef double s = _scale(fwhm, sigma_res)
    
    cdef object z = _pixels(s, n_scales)
    dbl_divide_inplace(z, s)
    
    cdef object k = np_zeros(z.shape[0], dtype=np_float64)
    cdef double k_sum = 0.0

    cdef Py_ssize_t i, n = z.shape[0]
    for i in range(n):
        k[i] = math_exp(-0.5 * z[i] * z[i])
        k_sum += k[i]

    dbl_multiply_inplace(k, 1.0 / (fwhm * k_sum))
    for i in range(n):
        k[i] *= (z[i] * z[i] - 1)

    return k

def kernel_deriv(
    const double fwhm,
    const double sigma_res,
    const double n_scales,
):
    return _kernel_deriv(fwhm, sigma_res, n_scales)

cdef object _convolve_signal(
    const double[::1] signal,
    const double[::1] kernel,
):
    cdef int l = kernel.shape[0] // 2
    return fftconvolve(
        np_pad(signal, pad_width=l, mode="edge"),
        kernel,
        mode="valid",
    )

def convolve_signal(
    const double[::1] signal,
    const double[::1] kernel,
):
    return _convolve_signal(signal, kernel)

cdef object _convolve_signal2d(
    const double[:,::1] signal,
    const double[::1] kernel,
):
    cdef int l = kernel.shape[0] // 2
    cdef object _kernel = np_zeros((1, kernel.shape[0]), dtype=np_float64)
    arr_add_inplace(_kernel[0, :], kernel)

    return fftconvolve(
        np_pad(signal, pad_width=((0, 0), (l, l)), mode="edge"),
        _kernel,
        mode="valid",
        axes=1,
    )

def convolve_signal2d(
    const double[:,::1] signal,
    const double[::1] kernel,
):
    return _convolve_signal2d(signal, kernel)

cdef int _identify_closest_idx(
    const double[::1] fwhm,
    const double fwhm_final,
):
    cdef int idx
    if fwhm_final >= fwhm[fwhm.shape[0] - 1]:
        idx = fwhm.shape[0] - 1
    else:
        idx = np_searchsorted(fwhm, fwhm_final, side="right") - 1

    return idx

def identify_closest_idx(
    const double[::1] fwhm,
    const double fwhm_final,
):
    return _identify_closest_idx(fwhm, fwhm_final)

cdef int _identify_closest_idx_for_deriv(
    const double[::1] fwhm,
    const double fwhm_final,
):
    cdef int idx = _identify_closest_idx(fwhm, fwhm_final)

    if idx != 0 and fwhm[idx] == fwhm_final:
        idx -= 1

    return idx

def identify_closest_idx_for_deriv(
    const double[::1] fwhm,
    const double fwhm_final,
):
    return _identify_closest_idx_for_deriv(fwhm, fwhm_final)

cdef object _convolve(
    const double[:,::1] data,
    const double[::1] fwhm,
    const double fwhm_final,
    const double sigma_res,
    const double n_scales,
):
    cdef int idx = _identify_closest_idx(fwhm, fwhm_final)
    cdef double fwhm_init = fwhm[idx]

    cdef object y = np_copy(data[idx, :])
    
    cdef double fwhm_kernel
    cdef object kernel

    if fwhm_init == fwhm_final:
        return y
    else:
        fwhm_kernel = math_sqrt(fwhm_final * fwhm_final - fwhm_init * fwhm_init)
        kernel = _kernel(fwhm_kernel, sigma_res, n_scales)
        return _convolve_signal(y, kernel)

def convolve(
    const double[:,::1] data,
    const double[::1] fwhm,
    const double fwhm_final,
    const double sigma_res,
    const double n_scales,
):
    return _convolve(data, fwhm, fwhm_final, sigma_res, n_scales)

cdef object _convolve_deriv(
    const double[:,::1] data,
    const double[::1] fwhm,
    const double fwhm_final,
    const double sigma_res,
    const double n_scales,
):
    cdef Py_ssize_t i, n = data.shape[1]
    cdef object dy

    cdef int idx
    cdef double fwhm_init, fwhm_kernel
    cdef object signal, kernel_deriv

    if fwhm_final == fwhm[0]:
        dy = np_copy(data[0, :])
        dbl_multiply_inplace(dy, -FWHM_TO_SIGMA)
        return dy
    else:
        idx = _identify_closest_idx_for_deriv(fwhm, fwhm_final)
        fwhm_init = fwhm[idx]
        signal = data[idx, :]
        
        fwhm_kernel = math_sqrt(fwhm_final * fwhm_final - fwhm_init * fwhm_init)
        kernel_deriv = _kernel_deriv(fwhm_kernel, sigma_res, n_scales)
        dbl_multiply_inplace(kernel_deriv, fwhm_final / fwhm_kernel)
                
        return _convolve_signal(signal, kernel_deriv)

def convolve_deriv(
    const double[:,::1] data,
    const double[::1] fwhm,
    const double fwhm_final,
    const double sigma_res,
    const double n_scales,
):
    return _convolve_deriv(data, fwhm, fwhm_final, sigma_res, n_scales)