from typing import Self
from abc import ABC, abstractmethod
from numpy import (
    log, isfinite, arange, empty, exp, median, full_like, float64, maximum,
    stack,
)
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from pathlib import Path

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_utils.setup import Info
from quasar_utils.binning import alpha_matrix_sparse, lin_dx
from quasar_utils.interpolation import create_interp_matrix

templates_dir: Path = Path(__file__).parent / 'templates'

def drop_nan(arr: NDArray[float64]) -> NDArray[float64]:
    return arr[isfinite(arr)]

def drop_nonpos(arr: NDArray[float64]) -> NDArray[float64]:
    return arr[isfinite(arr) & (arr > 0)]

SIGMA_TO_FWHM: float = 2 * (2 * log(2))**0.5
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM

TemplateTuple = tuple[FloatMatrix, FloatVector, FloatVector]

class BaseTemplate(ABC):
    SIGMA_TO_FWHM: float = 2 * (2 * log(2))**0.5
    FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM

    @validate_call
    def __init__(
        self,
        fwhm: FloatVector,
        x: FloatVector,
        data: FloatMatrix,
        *,
        info: Info = None,
        is_logspace: bool = False,
        name: str | None = 'no_name',
        path: AbsoluteFITSPath | None = None,
    ):
        assert fwhm.size == data.shape[0], "fwhm and data size mismatch"
        assert x.size == data.shape[1], "x and data size mismatch"

        self.fwhm: NDArray[float64] = fwhm
        self.x: NDArray[float64] = x
        self.data: NDArray[float64] = data

        self.info: Info = info
        self.is_logspace: bool = is_logspace
        self.name: str = name
        self.path: Path | None = path

        self._alpha_matrix: csr_matrix | None = None
        self._beta_matrix: csr_matrix | None = None
        self._xn: NDArray[float64] | None = None

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected {cls.__name__} instance, \
                got {type(value).__name__} instead."
            raise PydanticCustomError('validation_error', msg)
        return value
        
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return no_info_plain_validator_function(cls._validate)

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
    
    @abstractmethod
    def copy(self, with_matrices: bool = False) -> Self:
        """
        Creates a copy of the current Template. If `with_matrices` is True, the 
        logspace-transformation matrices are also copied, if available.
        """
        pass

    @abstractmethod
    def upsample(
        self,
        fwhm: FloatVector,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Upsamples the Template to the specified FWHM values.
        """
        pass

    @abstractmethod
    def resample(
        self,
        fwhm: FloatVector,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Resamples the Template to the specified FWHM values.
        """
        pass

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
        if self.is_logspace: 
            return self if inplace else self.copy(with_matrices=True)

        dx = lin_dx(self.x)
        x_edges = empty(self.x.size + 1, dtype=float)
        x_edges[:-1] = self.x - dx / 2
        x_edges[-1] = self.x[-1] + dx[-1] / 2

        sigma_res: float = self.info.loading['sigma_res']

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

        self._alpha_matrix = alpha_matrix_sparse.__wrapped__(
            x_edges, xr_edges,
        )
        self._beta_matrix = alpha_matrix_sparse.__wrapped__(
            xr_edges, xn_edges,
        )
        self._xn = 0.5 * (xn_edges[:-1] + xn_edges[1:])

        if inplace: 
            obj = self
        else:       
            obj = self.copy(with_matrices=True)

        obj.x = xr
        obj.data = maximum(self._alpha_matrix.dot(self.data.T).T, 0)
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
        """
        if inplace: 
            obj = self
        else:       
            obj = self.copy(with_matrices=True)

        M, b = create_interp_matrix(self.x, x, left=0, right=0)
        
        obj.data = maximum(stack([M.dot(y) + b for y in self.data], axis=0), 0)
        obj.x = x

        return obj
    
    # @validate_call
    # def save(
    #     self,
    #     path: AbsoluteFITSPath,
    #     overwrite: bool = False,
    # ) -> None:

    #     assert not (path.exists() and not overwrite), \
    #         f"File {path} already exists. Use overwrite=True to overwrite."

    #     v_unit: str = self.info.units['velocity_unit'].to_string()
    #     x_unit: str = self.info.units['wavelength_unit'].to_string()
    #     f_unit: str = self.info.units.getFluxUnit().to_string()

    #     hdul: fits.HDUList = fits.HDUList()

    #     hdul.append(fits.PrimaryHDU(
    #         data = self.info.units.getFlux(self.data).to(f_unit).value,
    #     ))
    #     hdul[0].header['NAME'] = self.name
    #     hdul[0].header['CTYPE1'] = ('fwhm', 'fwhm axis')
    #     hdul[0].header['CTYPE2'] =  ('x', 'spectral axis')
    #     hdul[0].header['BUNIT'] = (f_unit, 'flux unit')
    #     hdul[0].header['LOGSPACE'] = 'y' if self.is_logspace else 'n'

    #     if self.path is not None: 
    #         hdul[0].header['PATH'] = str(self.path)

    #     col_fwhm: fits.Column = fits.Column(
    #         name = 'fwhm',
    #         format = 'F',
    #         unit = v_unit,
    #         array = self.info.units.getC(self.fwhm).to(v_unit).value,
    #     )
    #     col_x: fits.Column = fits.Column(
    #         name = 'x',
    #         format = 'F',
    #         unit = x_unit,
    #         array = self.info.units.getWavelength(self.x).to(x_unit).value,
    #     )
    #     hdul.append(fits.BinTableHDU.from_columns([col_fwhm, col_x]))

    #     if hasattr(self, '_alpha_matrix'):
    #         col_alpha_data = fits.Column(
    #             name='alpha_data',
    #             format='D',
    #             array=self._alpha_matrix.data.flatten(),
    #         )
    #         col_alpha_indices = fits.Column(
    #             name='alpha_indices',
    #             format='K',
    #             array=self._alpha_matrix.indices,
    #         )
    #         col_alpha_indptr = fits.Column(
    #             name='alpha_indptr',
    #             format='K',
    #             array=self._alpha_matrix.indptr,
    #         )
    #         col_beta_data = fits.Column(
    #             name='beta_data',
    #             format='D',
    #             array=self._beta_matrix.data.flatten(),
    #         )
    #         col_beta_indices = fits.Column(
    #             name='beta_indices',
    #             format='K',
    #             array=self._beta_matrix.indices,
    #         )
    #         col_beta_indptr = fits.Column(
    #             name='beta_indptr',
    #             format='K',
    #             array=self._beta_matrix.indptr,
    #         )
    #         hdul.append(fits.BinTableHDU.from_columns([
    #             col_alpha_data, col_alpha_indices, col_alpha_indptr,
    #             col_beta_data, col_beta_indices, col_beta_indptr,
    #         ]))

    #         hdr = hdul[2].header
            
    #         # Alpha-matrix
    #         hdr['ASHAPE'] = (
    #             "{}/{}".format(*self._alpha_matrix.shape),
    #             'alpha matrix shape',
    #         )
    #         hdr['A_VAL'] = (
    #             self._alpha_matrix.data.size,
    #             "no. non-zero values",
    #         )
    #         hdr['A_IND'] = (
    #             self._alpha_matrix.indices.size,
    #             "no. indices",
    #         )
    #         hdr['A_PTR'] = (
    #             self._alpha_matrix.indptr.size,
    #             "no. index pointers",
    #         )
    #         # Beta-matrix
    #         hdr['BSHAPE'] = (
    #             "{}/{}".format(*self._beta_matrix.shape),
    #             'beta matrix shape',
    #         )
    #         hdr['B_VAL'] = (
    #             self._beta_matrix.data.size,
    #             "no. non-zero values",
    #         )
    #         hdr['B_IND'] = (
    #             self._beta_matrix.indices.size,
    #             "no. indices",
    #         )
    #         hdr['B_PTR'] = (
    #             self._beta_matrix.indptr.size,
    #             "no. index pointers",
    #         )

    #     hdul.writeto(
    #         path,
    #         overwrite = overwrite,
    #     )

    # @classmethod
    # @cache
    # def load(
    #     cls,
    #     path: str | AnyFITSPath,
    #     info: Info = None,
    # ) -> Self:
    #     _path: str = str(path)
    #     if not _path.endswith('.fits'):
    #         _path += '.fits'

    #     return _load(Path(_path), info)
    
class LoadedTemplates:
    template_files: dict[Path, str] = dict()
    templates: dict[str, BaseTemplate] = dict()

    def __class_getitem__(cls, template_name: Path | str) -> BaseTemplate:
        if isinstance(template_name, Path): 
            if template_name in cls.template_files.keys():
                return cls.templates[cls.template_files[template_name]]
            
            _path = template_name.name
        else:
            _path = template_name

        key: str = _path.removesuffix('.fits')
        if key in cls.templates.keys():
            return cls.templates[key]
        
        try:
            template: BaseTemplate = BaseTemplate.load(template_name)
            cls.template_files[template.path] = key
            cls.templates[key] = template
            return template
        
        except Exception as e:
            msg = f"Template not found: '{template_name}' due to: {e}"
            raise KeyError(msg)