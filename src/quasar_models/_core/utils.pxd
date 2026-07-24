cdef void dbl_add_inplace(double[::1] arr1, double value) noexcept nogil

cdef void dbl_subtract_inplace(double[::1] arr1, double value) noexcept nogil

cdef void dbl_multiply_inplace(double[::1] arr1, double value) noexcept nogil

cdef void dbl_divide_inplace(double[::1] arr1, double value) noexcept nogil

###

cdef void arr_add_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil

cdef void arr_subtract_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil

cdef void arr_multiply_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil

cdef void arr_divide_inplace(double[::1] arr1, const double[::1] arr2) noexcept nogil

###

cdef void multiply_and_add_to(double[::1] arr1, const double[::1] arr2, double value) noexcept nogil

cdef void multiply_and_multiply_to(double[::1] arr1, const double[::1] arr2, double value) noexcept nogil