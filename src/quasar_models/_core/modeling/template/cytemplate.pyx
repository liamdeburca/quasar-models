from libc.math cimport (
    sqrt as math_sqrt,
)
from numpy import (
    ascontiguousarray as np_ascontiguousarray,
    asarray as np_asarray,
    zeros as np_zeros,
    float64 as np_float64,
)
from quasar_models._core.modeling.split.evaluate cimport (
    _evaluate as _split_evaluate,
)
from quasar_models._core.modeling.split.fit_deriv cimport (
    _fit_deriv_only_split       as _split_fit_deriv_only_split,
    _fit_deriv_only_left        as _split_fit_deriv_only_left,
    _fit_deriv_only_right       as _split_fit_deriv_only_right,
    _fit_deriv_split_and_left   as _split_fit_deriv_split_and_left,
    _fit_deriv_split_and_right  as _split_fit_deriv_split_and_right,
    _fit_deriv_left_and_right   as _split_fit_deriv_left_and_right,
    _fit_deriv_all              as _split_fit_deriv_all,
)
from quasar_models._core.convolution.utils cimport (
    _kernel,
    _convolve_signal2d,
)
from quasar_models._core.utils cimport (
    arr_multiply_inplace,
    arr_add_inplace,
    multiply_and_multiply_to,
)

cdef class CyTemplate:
    """
    Cached template wrapper that holds typed memoryviews of template data.
    
    Memoryviews are created once during initialization, eliminating repeated
    buffer protocol validation overhead on each evaluate() call.
    
    Attributes
    ----------
    fwhm : double[::1]
        FWHM values as C-contiguous memoryview
    x : double[::1]
        Wavelength grid as C-contiguous memoryview
    data : double[:,::1]
        Template data (fwhm_bins, wavelength_bins) as C-contiguous memoryview
    sigma_res : double
        Velocity resolution
    """
    def __cinit__(
        self,
        double[::1] fwhm,    
        double[::1] x,
        double[:,::1] data,
        double sigma_res,
    ):
        self.fwhm = fwhm
        self.x = x
        self.data = data
        self.sigma_res = sigma_res

    def __reduce__(self):
        """Return state for pickling using a factory function."""
        return (
            _cytemplate_unpickle,
            (
                np_asarray(self.fwhm, dtype=np_float64, order="C"),
                np_asarray(self.x, dtype=np_float64, order="C"),
                np_asarray(self.data, dtype=np_float64, order="C"),
                self.sigma_res,
            ),
        )

    @staticmethod
    def fromTemplate(
        object python_template,
        bool simplify,
    ):
        """
        Factory method to create CyTemplate from a Python template object.
        
        Parameters
        ----------
        python_template : BaseTemplate
            A template instance with fwhm, x, data, sigma_res, 
            and normalisation attributes.
        simplify : bool
            Whether to simplify the template by constructing a CyTemplate using 
            only the smallest fwhm value. 
        
        Returns
        -------
        CyTemplate
            A new CyTemplate instance with cached memoryviews.
        """
        if simplify:
            fwhm = python_template.fwhm[:1]
            x = python_template.x
            data = python_template.data[:1,:]
        else:
            fwhm = python_template.fwhm
            x = python_template.x
            data = python_template.data
        
        return CyTemplate(
            np_ascontiguousarray(fwhm),
            np_ascontiguousarray(x),
            np_ascontiguousarray(data / python_template.normalisation),
            python_template.sigma_res,
        )

    cdef CyTemplate split(
        self,
        const double split,
        const double left,
        const double right,
        const double scale,
    ):
        cdef Py_ssize_t n = len(self.x)

        cdef double[:,::1] split_data = np_zeros((1, n), dtype=np_float64)
        _split_evaluate(split_data[0, :], self.x, split, left, right, self.sigma_res, scale)
        arr_multiply_inplace(split_data[0, :], self.data[0, :])

        return CyTemplate(self.fwhm[0:1], self.x, split_data, self.sigma_res)

    ### Split derivatives

    cdef void _add_split_fit_deriv(
        self,
        double[:,::1] derivs,
        double[:,::1] split_derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
        const int[:] indices,
    ):
        cdef int i, idx, m = indices.shape[0]
        cdef double[::1] kernel
        cdef double fwhm_kernel, fwhm_init
            
        # Apply flux scaling and data multiplication
        for i in range(m):
            idx = indices[i]
            multiply_and_multiply_to(split_derivs[idx, :], self.data[0, :], flux)
        
        # Handle fwhm convolution if needed
        fwhm_init = self.fwhm[0]
        if fwhm_init != fwhm:
            fwhm_kernel = math_sqrt(fwhm * fwhm - fwhm_init * fwhm_init)
            kernel = _kernel(fwhm_kernel, self.sigma_res, n_scales)
            split_derivs = _convolve_signal2d(split_derivs, kernel)

        for i in range(m):
            idx = indices[i]
            arr_add_inplace(derivs[idx, :], split_derivs[idx, :])

    cdef void add_split_fit_deriv_only_split(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[1] indices = [0]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_only_split(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )

    cdef void add_split_fit_deriv_only_left(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[1] indices = [1]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_only_left(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )

    cdef void add_split_fit_deriv_only_right(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[1] indices = [2]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_only_right(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )

    cdef void add_split_fit_deriv_split_and_left(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[2] indices = [0, 1]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_split_and_left(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )

    cdef void add_split_fit_deriv_split_and_right(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[2] indices = [0, 2]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_split_and_right(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )

    cdef void add_split_fit_deriv_left_and_right(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[2] indices = [1, 2]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_left_and_right(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )

    cdef void add_split_fit_deriv_all(
        self,
        double[:,::1] derivs,
        const double flux,
        const double fwhm,
        const double split,
        const double left,
        const double right,
        const double scale,
        const double n_scales,
    ):
        cdef Py_ssize_t n = self.x.shape[0]
        cdef double[:,::1] split_derivs
        cdef int[3] indices = [0, 1, 2]

        if flux != 0.0:
            # Compute split derivatives
            split_derivs = np_zeros((3, n), dtype=np_float64)
            _split_fit_deriv_all(
                split_derivs,
                self.x,
                split, left, right,
                self.sigma_res, scale,
            )
            self._add_split_fit_deriv(
                derivs,
                split_derivs,
                flux, fwhm, split, left, right, scale, n_scales,
                indices,
            )


def _cytemplate_unpickle(fwhm, x, data, sigma_res):
    """Factory function for unpickling CyTemplate objects."""
    return CyTemplate(
        np_ascontiguousarray(fwhm, dtype=np_float64),
        np_ascontiguousarray(x, dtype=np_float64),
        np_ascontiguousarray(data, dtype=np_float64),
        sigma_res,
    )