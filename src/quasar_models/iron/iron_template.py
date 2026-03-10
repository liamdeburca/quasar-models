from typing import Self
from pathlib import Path
from numpy import empty, float64, searchsorted
from functools import cache

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from .utils import _split_evaluate
from ..utils.template import BaseTemplate

from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_typing.pathlib import AnyFITSPath, AbsoluteFITSPath
from quasar_utils.setup import Info
from quasar_utils.convolution import convolve_signal, kernel

_this_file: Path = Path(__file__).resolve()

#Place cache in project root directory, in a hidden folder named '.cache/iron_templates'
PATH_TO_CACHE: Path = _this_file.parents[3] / '.cache/iron_templates'
if not PATH_TO_CACHE.exists(): PATH_TO_CACHE.mkdir(parents=True)

class IronTemplate(BaseTemplate):
    """
    Template class specifically designed for Iron pseudo-continua.
    """
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
        path: AnyFITSPath | None = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        super().__init__(
            fwhm, x, data,
            info=info, is_logspace=is_logspace,
            name=name, path=path,
        )

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, IronTemplate):
            msg = f"Expected an IronTemplate instance, \
                got {type(value)} instead."
            raise PydanticCustomError('validation_error', msg)
        return value
        
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return no_info_plain_validator_function(cls._validate)
    
    def copy(self, with_matrices: bool = False) -> Self:
        """
        Creates a copy of the current IronTemplate instance. If `with_matrices`
        is True, the logspace-transformation matrices are also copied, if 
        available.
        """
        temp: IronTemplate = IronTemplate(
            self.fwhm.copy(), self.x.copy(), self.data.copy(),
            split=self.split, left=self.left, right=self.right, 
            scale=self.scale, info=self.info, is_logspace=self.is_logspace,
            name=self.name, path=self.path,
        )
        if with_matrices and getattr(self, '_alpha_matrix', None) is not None:
            temp._alpha_matrix = self._alpha_matrix
            temp._beta_matrix = self._beta_matrix
            temp._xn = self._xn

        return temp
    
    def upsample(
        self,
        fwhm: FloatVector,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Upsamples the IronTemplate to the specified FWHM values.
        """        
        assert not (fwhm < self.fwhm[0]).any(), \
            "FWHM values must be >= current template's minimum FWHM."
        
        if self.is_logspace:
            if inplace: obj = self
            else:       obj = self.copy(with_matrices=True)

            data = empty(shape=(fwhm.size, self.x.size), dtype=float64)
            indices = searchsorted(self.fwhm, fwhm)

            for i, _fwhm in enumerate(fwhm):
                # Check if exact match exists at the insertion index
                if indices[i] < self.fwhm.size and self.fwhm[indices[i]] == _fwhm:
                    data[i,:] = self.data[indices[i]]
                else:
                    k = kernel(
                        (_fwhm**2 - self.fwhm[0]**2)**0.5,
                        self.info.loading['sigma_res'],
                    )
                    data[i,:] = convolve_signal.__wrapped__(self.data[0], k)

            obj.fwhm = fwhm
            obj.data = data
            
        else:
            template = self.createLogspace(inplace=False, keep_x=keep_x)
            template.upsample(fwhm, inplace=True)
            obj = self.mimicLogspace(template, inplace=inplace)

        return obj
    
    def resample(
        self,
        fwhm: FloatVector,
        inplace: bool = False,
        keep_x: bool = False,
    ) -> Self:
        """
        Resamples the IronTemplate to the specified FWHM values.
        """
        if inplace: obj = self
        else:       obj = self.copy(with_matrices=True)

        obj.fwhm = obj.fwhm[:1]
        obj.data = obj.data[:1]

        return obj.upsample(fwhm, inplace=True, keep_x=keep_x)

    def applySplit(
        self,
        split: float,
        left: float,
        right: float,
        scale: float,
        inplace: bool = False,
    ) -> Self:
        assert self.is_logspace, "Template must be in logspace."

        if inplace: obj: IronTemplate = self
        else:       obj: IronTemplate = self.copy(with_matrices=True)

        obj.split = split
        obj.left  = 1.0
        obj.right = 1.0
        obj.scale = scale

        obj.data *= self._get_split_weight(
            obj.x, split, left, right, scale,
        )[None,:]

        return obj
    
    def _get_split_weight(
        self,
        x: FloatVector,
        split: float,
        left: float,
        right: float,
        scale: float,
    ) -> FloatVector:
        """
        Calculates the split weight vector for the given parameters.

        Notes
        -----
        This method assumes that the x-array is in logspace (logbinned).
        """
        return _split_evaluate(
            x, split, left, right,
            sigma_res=self.info.loading['sigma_res'], scale=scale,
        )

    def save(
        self,
        path: str | AbsoluteFITSPath,
        overwrite: bool = False,
    ) -> AbsoluteFITSPath:
        """
        Saves the IronTemplate to a FITS file.
        """
        from .io import _save

        path: Path = Path(str(path).removesuffix('.fits') + '.fits')
        if not path.is_absolute():
            path = self.PATH_TO_CACHE / path.name

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
    
        return _save(self, path)

    @classmethod
    @cache
    def load(
        cls,
        path: str | AbsoluteFITSPath,
        info: Info = None,
    ) -> Self:
        """
        Loads an IronTemplate from a FITS file.
        """
        from .io import _load

        path: Path = Path(str(path).removesuffix('.fits') + '.fits')
        if not path.is_absolute():
            path = cls.PATH_TO_CACHE / path.name

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return _load(path, info=info)