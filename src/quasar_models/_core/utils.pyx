cdef void dbl_add_inplace(double[::1] arr1, double value) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] += value

cdef void dbl_subtract_inplace(double[::1] arr1, double value) noexcept nogil:
    dbl_add_inplace(arr1, -value)

cdef void dbl_multiply_inplace(double[::1] arr1, double value) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] *= value

cdef void dbl_divide_inplace(double[::1] arr1, double value) noexcept nogil:
    dbl_multiply_inplace(arr1, 1.0 / value)

###

cdef void arr_add_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] += arr2[i]

cdef void arr_subtract_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] -= arr2[i]

cdef void arr_multiply_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] *= arr2[i]

cdef void arr_divide_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] /= arr2[i]

###

cdef void multiply_and_add_to(double[::1] arr1, const double[::1] arr2, double value) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] += arr2[i] * value

cdef void multiply_and_multiply_to(double[::1] arr1, const double[::1] arr2, double value) noexcept nogil:
    cdef Py_ssize_t i, n = arr1.shape[0]
    for i in range(n):
        arr1[i] *= arr2[i] * value