__all__ = ['BaseTemplate']

from typing import Self
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

@dataclass
class BaseTemplate(ABC):
    fwhm: SortedFloatVector = field(kw_only=True)
    x: SortedFloatVector = field(kw_only=True)
    data: FloatMatrix = field(kw_only=True)

    is_logspace: bool = field(kw_only=True)
    sigma_res: float | None = field(default=None, kw_only=True)
    name: str = field(kw_only=True)

    path: AbsoluteFITSPath | None = field(default=None, kw_only=True)

    _alpha_matrix: csr_matrix_ | None = field(default=None, repr=False, kw_only=True)
    _beta_matrix: csr_matrix_ | None = field(default=None, repr=False, kw_only=True)
    _xn: SortedFloatVector | None = field(default=None, repr=False, kw_only=True)

    x_norm: float = field(kw_only=True)
    fwhm_norm: float = field(kw_only=True)
    normalisation: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.is_logspace:
            assert self.sigma_res is not None
        else:
            assert self.sigma_res is None

        self._validate_shapes()

    def __getstate__(self) -> dict:
        # Serialisation is primarily used in fitting routines, where binning 
        # matrices are not needed.
        return {
            'fwhm': self.fwhm,
            'x': self.x,
            'data': self.data,
            'is_logspace': self.is_logspace,
            'sigma_res': self.sigma_res,
            'name': self.name,
            'path': self.path,
            'x_norm': self.x_norm,
            'fwhm_norm': self.fwhm_norm,
            'normalisation': self.normalisation,
        }
    
    def __setstate__(self, state: dict) -> None:
        self.__init__(**state)
        
    def _validate_shapes(self):
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

    def __eq__(self, other: object) -> bool:
        """
        True if all attributes are equal. 
        """
        return isinstance(other, self.__class__) \
            and (self.name == other.name) \
            and (self.is_logspace == other.is_logspace) \
            and (self.sigma_res == other.sigma_res) \
            and (self.path == other.path) \
            and (self.x_norm == other.x_norm) \
            and (self.fwhm_norm == other.fwhm_norm) \
            and (self.normalisation == other.normalisation) \
            and array_equal(self.fwhm, other.fwhm) \
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
        *,
        sigma_res: float | None = None,
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
                        self.sigma_res,
                    )
                    data[i,:] = convolve_signal.__wrapped__(data_prev, k)

                fwhm_prev = fwhm_curr
                data_prev = data[i,:]

            obj.fwhm = fwhm
            obj.data = data
        else:
            assert sigma_res is not None

            template = self.createLogspace(
                sigma_res=sigma_res,
                keep_x=keep_x,
            )
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
        *,
        sigma_res: float,
        xr: FloatVector | None = None,
        keep_x: bool = False,
        conserve: bool = True,
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
        if self.is_logspace:
            return self.interpolate(xr, inplace=False)
        
        dx = lin_dx(self.x)
        x_edges = empty(self.x.size + 1, dtype=float)
        x_edges[:-1] = self.x - dx / 2
        x_edges[-1] = self.x[-1] + dx[-1] / 2

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

        cp = self.copy()
        cp.is_logspace = True
        cp.sigma_res = sigma_res
        cp._xn = 0.5 * (xn_edges[:-1] + xn_edges[1:])
        cp.x = xr

        self._alpha_matrix = cp._alpha_matrix = alpha_matrix_sparse.__wrapped__(
            x_edges, xr_edges,
            dx=dx,
            dxr=dxr,
            conserve=conserve,
        )
        self._beta_matrix = cp._beta_matrix = alpha_matrix_sparse.__wrapped__(
            xr_edges, xn_edges,
            dx=dxr,
            dxr=dx if keep_x else diff(xn_edges),
            conserve=conserve,
        )
        cp.data = maximum(cp._alpha_matrix.dot(cp.data.T).T, 0)

        cp.normalisation = None
        cp.__post_init__()

        return cp
    
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

        assert self._beta_matrix is not None, \
            "Current template must have a beta matrix."
        assert template._beta_matrix is not None, \
            "Provided template must have a beta matrix."
        assert self._beta_matrix is template._beta_matrix, \
            "Templates must share the same beta matrix if 'inplace=True'."

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
        