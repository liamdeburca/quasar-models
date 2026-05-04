from typing import Self, Literal
from pathlib import Path
from numpy import empty, stack, float64
from numpy.typing import NDArray

from pydantic import validate_call

from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_typing.pathlib import AnyAbsoluteFITSPath, AbsoluteFITSPath
from quasar_utils.setup import Info
from quasar_utils.convolution import convolve_signal, kernel

from .io import _load, _save
from .utils import get_x_grid
from ..utils.template import BaseTemplate

_this_file: Path = Path(__file__).resolve()

#Place cache in project root directory, in a hidden folder named '.cache/balmer_templates'
PATH_TO_CACHE: Path = _this_file.parents[3] / '.cache/balmer_templates'
if not PATH_TO_CACHE.exists(): 
    PATH_TO_CACHE.mkdir(parents=True)

class BalmerTemplate(BaseTemplate):
    """
    Template class specifically designed for Balmer pseudo-continua.
    """
    PATH_TO_CACHE: Path = PATH_TO_CACHE

    @validate_call
    def __init__(
        self,
        fwhm: FloatVector,
        x: FloatVector,
        data: FloatMatrix,
        *,
        edge: float | None = None,
        temp: float | None = None,
        dens: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        ratio: float | None = None,
        waves: FloatVector | None = None,
        weights: FloatVector | None = None,
        case: Literal['A', 'B'] | None = None,
        n_l: int | None = None,
        n_u: int | None = None,
        info: Info = None,
        is_logspace: bool = False,
        based_on_template: bool = False,
        name: str | None = None,
        path: AbsoluteFITSPath | None = None,
    ):
        """
        **PYDANTIC VALIDATED FUNCTION**
        """
        super().__init__(
            fwhm, x, data,
            info=info, is_logspace=is_logspace, name=name, path=path,
        )
        self.based_on_template: bool = based_on_template

        if not based_on_template:
            assert edge is not None, "edge must be provided"
            assert temp is not None, "temp must be provided"
            assert dens is not None, "dens must be provided"
            assert tau is not None, "tau must be provided"
            assert scale is not None, "scale must be provided"
            assert ratio is not None, "ratio must be provided"

            assert waves is not None, "waves must be provided"
            assert weights is not None, "weights must be provided"
            assert case is not None, "case must be provided"
            assert n_l is not None, "n_l must be provided"
            assert n_u is not None, "n_u must be provided"

            weights /= weights.sum()
    
        self.edge: float | None = edge
        self.temp: float | None = temp
        self.dens: float | None = dens
        self.tau: float | None = tau
        self.scale: float | None = scale
        self.ratio: float | None = ratio

        self.waves: FloatVector | None = waves
        self.weights: FloatVector | None = weights
        self.case: Literal['A', 'B'] | None = case
        self.n_l: int | None = n_l
        self.n_u: int | None = n_u
    
    def copy(self, with_matrices: bool = False) -> Self:
        """
        Creates a copy of the current BalmerTemplate instance. If 
        `with_matrices` is True, the logspace-transformation matrices are also 
        copied, if available.
        """
        temp: BalmerTemplate = BalmerTemplate(
            self.fwhm.copy(), self.x.copy(), self.data.copy(),
            edge=self.edge, temp=self.temp, dens=self.dens, 
            tau=self.tau, scale=self.scale, ratio=self.ratio, 
            waves=self.waves.copy(), weights=self.weights.copy(), 
            case=self.case, n_l=self.n_l, n_u=self.n_u,
            info=self.info, based_on_template=self.based_on_template, 
            is_logspace=self.is_logspace, path=self.path, name=self.name,
        )
        if with_matrices and getattr(self, '_alpha_matrix', None):
            temp._alpha_matrix = self._alpha_matrix
            temp._beta_matrix = self._beta_matrix
            temp._xn = self._xn

        return temp

    def upsample(
        self,
        fwhm: FloatVector,
        x: FloatVector | None = None,
        inplace: bool = False,
    ) -> Self:
        """
        Upsamples the BalmerTemplate to the specified FWHM- and x-values.
        """
        from .evaluation import evaluate

        if not self.based_on_template:
            if x is None: 
                x = self.x

            sigma_res = self.info.loading['sigma_res']
            boltz = self.info.units.getBoltzmannFactor()

            def inner(fwhm: float) -> NDArray[float64]:
                return evaluate(
                    x,
                    1.0, fwhm,
                    self.temp, self.tau, self.scale, self.ratio,
                    sigma_res=sigma_res, 
                    edge=self.edge,
                    waves=self.waves, 
                    weights=self.weights, 
                    boltz=boltz,
                    x_grid=get_x_grid(self.edge.value, sigma_res)
                )
            
            if inplace: 
                obj = self
            else:       
                obj = self.copy()

            obj.fwhm = fwhm
            obj.x = x
            obj.data = stack([inner(_fwhm) for _fwhm in fwhm], axis=0)

        else:
            assert not (fwhm < self.fwhm[0]).any(), \
                "FWHM values must be >= current template's minimum FWHM."
            
            if self.is_logspace:
                if inplace: 
                    obj = self
                else:       
                    obj = self.copy(with_matrices=True)

                data = empty(
                    shape=(fwhm.size, self.x.size), 
                    dtype=float,
                )

                for i, _fwhm in enumerate(fwhm):
                    if _fwhm in self.fwhm:
                        data[i,:] = self.data[self.fwhm == _fwhm]
                        continue

                    k = kernel(
                        (_fwhm**2 - self.fwhm[0]**2)**0.5,
                        self.info.loading['sigma_res'],
                    )
                    data[i,:] = convolve_signal(self.data[0], k)

                obj.fwhm = fwhm
                obj.data = data

            else:
                template = self.createLogspace(inplace=False, keep_x=True)
                template.upsample(fwhm, inplace=True)
                obj = self.mimicLogspace(template, inplace=inplace)

            if x is not None:
                obj.interpolate(x, inplace=True)
        
        return obj
    
    def resample(
        self,
        fwhm: FloatVector,
        x: FloatVector | None = None,
        inplace: bool = False,
    ) -> Self:
        """
        Resamples the BalmerTemplate to the specified FWHM- and x-values.
        """        
        if inplace: 
            obj = self
        else:       
            obj = self.copy()

        if self.based_on_template:
            # Remove entries to force re-evaluation
            obj.fwhm = obj.fwhm[:1]
            obj.data = obj.data[:1,:]
        else:
            # No need to remove entries, attributes are redefined anyway
            pass

        return obj.upsample(fwhm, x=x, inplace=True)

    ### I/O
    
    def save(
        self,
        path: AnyAbsoluteFITSPath,
        overwrite: bool = False,
    ) -> AbsoluteFITSPath:
        """
        Saves the BalmerTemplate to a FITS file.
        """
        path: Path = Path(str(path).removesuffix('.fits') + '.fits')
        if not path.is_absolute():
            path = self.PATH_TO_CACHE / path.name

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        
        _save(self, path)

        return path

    @classmethod
    def load(
        cls,
        path: AbsoluteFITSPath,
        info: Info = None,
    ) -> Self:
        """
        Loads an BalmerTemplate from a FITS file.
        """
        path: Path = Path(str(path).removesuffix('.fits') + '.fits')
        if not path.is_absolute():
            path = cls.PATH_TO_CACHE / path.name

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return _load(path, info=info)