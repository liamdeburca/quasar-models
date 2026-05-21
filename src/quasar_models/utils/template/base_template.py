__all__ = ['BaseTemplate']

from typing import Self, ClassVar, Any
from abc import ABC, abstractmethod
from numpy import (
    log, isfinite, arange, empty, exp, median, full_like, float64, maximum,
    stack, searchsorted, array_equal, diff, 
)
from numpy.typing import NDArray
from pathlib import Path

from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.numpy import FloatVector, FloatMatrix, SortedFloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.bounds import AstropyBounds

from quasar_utils.setup import Info
from quasar_utils.binning import alpha_matrix_sparse, lin_dx
from quasar_utils.interpolation import create_interp_matrix
from quasar_utils.convolution import convolve_signal, kernel
from quasar_utils.raster import rasterise
from quasar_utils.decorators import validate_call

templates_dir: Path = Path(__file__).parent / 'templates'

def drop_nan(arr: NDArray[float64]) -> NDArray[float64]:
    return arr[isfinite(arr)]

def drop_nonpos(arr: NDArray[float64]) -> NDArray[float64]:
    return arr[isfinite(arr) & (arr > 0)]

SIGMA_TO_FWHM: float = 2 * (2 * log(2))**0.5
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM

TemplateTuple = tuple[FloatMatrix, FloatVector, FloatVector]

@dataclass(eq=False)
class BaseTemplate(ABC):
    fwhm: SortedFloatVector
    x: SortedFloatVector
    data: FloatMatrix

    info: Info = field(default_factory=Info, kw_only=True)
    is_logspace: bool = field(default=False, kw_only=True)
    name: str = field(default='no_name', kw_only=True)
    path: AbsoluteFITSPath | None = field(default=None, kw_only=True)

    _alpha_matrix: csr_matrix_ | None = field(default=None, repr=False, kw_only=True)
    _beta_matrix: csr_matrix_ | None = field(default=None, repr=False, kw_only=True)
    _xn: SortedFloatVector | None = field(default=None, repr=False, kw_only=True)

    SIGMA_TO_FWHM: ClassVar[float] = 2 * (2 * log(2))**0.5
    FWHM_TO_SIGMA: ClassVar[float] = 1 / SIGMA_TO_FWHM

    def __post_init__(self):
        if self.fwhm.size != self.data.shape[0]:
            msg = "fwhm size {} does not match first axis of data {}".format(
                self.fwhm.size, self.data.shape[0],
            )
            raise ValueError(msg)
        if self.x.size != self.data.shape[1]:
            msg = "x size {} does not match second axis of data {}".format(
                self.x.size, self.data.shape[1],
            )
            raise ValueError(msg)
        
    def __getitem__(self, *sel) -> Self:
        """
        Create a copy of the Template based on the selection.
        """
        if len(sel) > 2:
            raise IndexError("Too many indices for Template.")
        
        obj = self.copy(with_matrices=False)
        if len(sel) == 1:
            obj.data = obj.data[sel[0],:]
            obj.fwhm = obj.fwhm[sel[0]]

        else:
            obj.data = obj.data[sel[0],:][:,sel[1]]
            obj.fwhm = obj.fwhm[sel[0]]
            obj.x    = obj.x   [sel[1]]

        return obj

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseTemplate):
            return False

        return array_equal(self.fwhm, other.fwhm) \
            and array_equal(self.x, other.x) \
            and array_equal(self.data, other.data)
    
    def normalise(self, inplace: bool = False) -> Self:
        obj = self if inplace else self.copy(with_matrices=True)

        if hasattr(self, 'normalisation') \
            and self.normalisation is not None \
            and self.normalisation != 0:
            obj.data /= self.normalisation
            obj.normalisation = 1.0

        return obj
    
    @abstractmethod
    def copy(self, with_matrices: bool = False) -> Self:
        """
        Creates a copy of the current Template. If `with_matrices` is True, the 
        logspace-transformation matrices are also copied, if available.
        """
        pass

    def upsample(
        self,
        fwhm: SortedFloatVector,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Upsamples the Template to the specified FWHM values.
        """        
        if self.is_logspace:
            obj = self if inplace else self.copy(with_matrices=True)

            data = empty(shape=(fwhm.size, self.x.size), dtype=float64)
            indices = searchsorted(self.fwhm, fwhm)

            fwhm_prev = self.fwhm[0]
            data_prev = self.data[0,:]

            for i, fwhm_curr in enumerate(fwhm):
                # Check if exact match exists at the insertion index
                if indices[i] < self.fwhm.size \
                    and self.fwhm[indices[i]] == fwhm_curr:
                    data[i,:] = self.data[indices[i]]
                else:
                    k = kernel(
                        (fwhm_curr**2 - fwhm_prev**2)**0.5,
                        self.info.loading.sigma_res,
                    )
                    data[i,:] = convolve_signal.__wrapped__(data_prev, k)

                fwhm_prev = fwhm_curr
                data_prev = data[i,:]

            obj.fwhm = fwhm
            obj.data = data
        else:
            template = self.createLogspace(inplace=False, keep_x=keep_x)
            template.upsample(fwhm, inplace=True)
            obj = self.mimicLogspace(template, inplace=inplace)

        return obj

    def resample(
        self,
        fwhm: SortedFloatVector,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Resamples the Template to the specified FWHM values.
        """
        obj = self if inplace else self.copy(with_matrices=True)
        obj.fwhm = obj.fwhm[:1]
        obj.data = obj.data[:1]
        return obj.upsample(fwhm, inplace=True, keep_x=keep_x)

    ### I/O: Abstract methods ###

    @abstractmethod
    def save(
        self,
        path: AbsoluteFITSPath,
        overwrite: bool = False,
    ) -> AbsoluteFITSPath:
        """
        Saves the Template to a FITS file.
        """
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: AbsoluteFITSPath,
        info: Info | None = None,
    ) -> Self:
        pass

    ### Space transformations ###

    def createLogspace(
        self,
        xr: FloatVector | None = None,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Creates a logspace equivalent of the current Template. 

        Parameters
        ----------
        xr : FloatVector | None
            The new logspace coordinates. If None, they will be generated
            based on `sigma_res`.
        inplace : bool, optional
            Whether to modify the current Template or return a new one.
            Default is False.

        Returns
        -------
        template : Template
            The logspace-equivalent template.
        """
        obj = self if inplace else self.copy(with_matrices=False)
        if self.is_logspace:
            return (
                obj 
                if array_equal(obj.x, xr) else 
                obj.interpolate(xr, inplace=True)
            )

        dx = lin_dx(obj.x)
        x_edges = empty(obj.x.size + 1, dtype=float)
        x_edges[:-1] = obj.x - dx / 2
        x_edges[-1] = obj.x[-1] + dx[-1] / 2

        sigma_res: float = obj.info.loading['sigma_res']

        if xr is None:
            nr = log(x_edges[-1] / x_edges[0]) // log(1 + sigma_res) + 1
            logxr_edges = log(x_edges[0]) + sigma_res * arange(nr + 1)
            xr_edges = exp(logxr_edges)
            xr = exp(0.5 * (logxr_edges[:-1] + logxr_edges[1:]))
        else:
            logxr = log(xr)
            dlogxr = full_like(xr, fill_value=sigma_res)
            logxr_edges = empty(xr.size + 1, dtype=float)
            logxr_edges[:-1] = logxr - dlogxr / 2 
            logxr_edges[-1] = logxr[-1] + dlogxr[-1] / 2
            xr_edges = exp(logxr_edges)

        dxr = xr * sigma_res

        if keep_x:
            xn_edges = x_edges
        else:
            dxn = median(dx)
            n_left =  int(abs(x_edges[0] - xr_edges[0]) // dxn + 1)
            n_right = int(abs(xr_edges[-1] - x_edges[-1]) // dxn + 1)

            xn_edges = empty(int(n_left + len(x_edges) + n_right))
            xn_edges[:n_left] = x_edges[0] + dxn * arange(-n_left, 0)
            xn_edges[n_left:-n_right] = x_edges
            xn_edges[-n_right:] = x_edges[-1] + dxn * arange(1, n_right+1)

        obj._alpha_matrix = alpha_matrix_sparse.__wrapped__(
            x_edges, xr_edges,
            dx=dx,
            dxr=dxr,
            conserve=obj.info.loading.conserve,
        )
        obj._beta_matrix = alpha_matrix_sparse.__wrapped__(
            xr_edges, xn_edges,
            dx=dxr,
            dxr=dx if keep_x else diff(xn_edges),
            conserve=obj.info.loading.conserve,
        )
        obj._xn = 0.5 * (xn_edges[:-1] + xn_edges[1:])

        obj.x = xr
        obj.data = maximum(obj._alpha_matrix.dot(obj.data.T).T, 0)
        obj.is_logspace = True

        return obj
    
    def mimicLogspace(
        self,
        template: Self,
        inplace: bool = False,
    ) -> Self:
        """
        Mimics the logspace-equivalent of this template, i.e. the two templates 
        must share identical beta matrices (logspace -> any space). 
        
        This method updates the fwhm and data arrays using the 
        logspace-equivalent template, and is therefore useful for when the 
        logspace-equivalent templates has been upsampled/resampled.

        Parameters
        ----------
        template : Template
            The logspace-equivalent template to mimic.
        inplace : bool, optional
            Whether to modify the current template or a copy.
            Default is False.
        """
        from numpy import maximum

        assert template.is_logspace, "Provided template must be in logspace."
        assert not self.is_logspace, "Current template must be in linspace."

        assert hasattr(self, '_beta_matrix'), \
            "Current template must have a beta matrix."
        assert hasattr(template, '_beta_matrix'), \
            "Provided template must have a beta matrix."
        assert self._beta_matrix is template._beta_matrix, \
            "Templates must share the same beta matrix."

        if inplace: 
            obj = self
        else:       
            obj = self.copy(with_matrices=True)

        obj.fwhm = template.fwhm.copy()
        obj.x = template._xn.copy()
        obj.data = maximum(self._beta_matrix.dot(template.data.T).T, 0)

        # Delete transformation matrices as x and data may have been transformed
        # differently
        del obj._alpha_matrix
        del obj._beta_matrix
        del obj._xn

        return obj

    def interpolate(
        self,
        x: FloatVector,
        inplace: bool = False,
    ) -> Self:
        """
        Interpolates the template to match the new x coordinates.

        If the new x coordinates match the current x coordinates, the same 
        template is returned (or a copy if `inplace=False`).
        """
        obj = self if inplace else self.copy(with_matrices=True)

        if array_equal(x, self.x):
            return obj

        M, b = create_interp_matrix(self.x, x, left=0, right=0)
        
        obj.data = maximum(stack([M.dot(y) + b for y in self.data], axis=0), 0)
        obj.x = x

        return obj
    
    @validate_call
    def rasterise(
        self,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        *,
        flux_bounds: AstropyBounds = (None, None),
        fwhm_bounds: AstropyBounds = (None, None),
    ) -> tuple[FloatVector, FloatVector]:
        """
        Performs a raster fit of the template to the provided data returning the 
        chi-square and flux value for each FWHM.
        """
        obj = self.interpolate(x, inplace=False)
        return rasterise.__wrapped__(
            y, dy, obj.fwhm, obj.data, 
            flux_bounds=flux_bounds, 
            fwhm_bounds=fwhm_bounds,
        )
        