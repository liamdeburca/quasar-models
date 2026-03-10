"""
AstroPy compatible model: BalmerModel.
"""
from typing import Self, Literal
from numpy import array, linspace, clip, stack, inf, argmin, float64
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from astropy.units import Unit
from astropy.modeling import Parameter
from functools import partial

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from . import evaluation
from .balmer_template import BalmerTemplate
from .utils import get_x_grid
from ..utils.basemodel import BaseModel
from ..continuum import PowerLawModel

from quasar_utils.setup import Info
from quasar_utils.raster import rasterise
from quasar_typing.numpy import FittableFloatVector

class BalmerModel(BaseModel):
    flux  = Parameter(default=1.0, min=0.0)
    fwhm  = Parameter(default=0,   min=0.0)
    temp  = Parameter(default=15_000.0, fixed=True)
    tau   = Parameter(default=1.0, fixed=True)
    scale = Parameter(default=3.0, fixed=True)
    ratio = Parameter(default=0.3, fixed=True)

    @validate_call(validate_return=False)
    def __init__(
        self,
        flux: float,
        fwhm: float,
        temp: float,
        tau: float,
        scale: float,
        ratio: float,
        *,
        edge: float = None,
        sigma_res: float = None,
        waves: NDArray[float64] = None,
        weights: NDArray[float64] = None,
        boltz: float = None,
        template: BalmerTemplate | None = None,
        name: str = 'balmer_pseudo_continuum',
        **kwargs,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **

        Notes
        -----
        The keyword arguments `edge`, `sigma_res`, `waves`, `weights`, and 
        `boltz` must be specified. Otherwise, a ValidationError is raised. 
        """
        super().__init__(
            flux, fwhm, temp, tau, scale, ratio,
            name=name, **kwargs,
        )

        self.edge: float = edge
        self.sigma_res: float = sigma_res
        self.waves: NDArray[float64] = waves
        self.weights: NDArray[float64] = weights / weights.sum()
        self.boltz: float = boltz

        # Calculate x_grid: wavelength grid from ~1000 Å to ~4000 Å
        self.x_grid: NDArray[float64] = get_x_grid(edge, sigma_res)

        # Template-fitting
        self.template: BalmerTemplate | None = template
        if template is not None:
            # Template-fitting enabled!
            self.temp.fixed  = True
            self.tau.fixed   = True
            self.scale.fixed = True
            self.ratio.fixed = True

    @property
    def template_tuple(self) -> tuple[NDArray[float64]] | None:
        if self.template is None: return None
        return (self.template.data, self.template.fwhm, self.template.x)
    
    @property
    def _perform_template_fitting(self) -> bool:
        return (self.template is not None) \
            and self.temp.fixed \
            and self.tau.fixed \
            and self.scale.fixed \
            and self.ratio.fixed

    def evaluate(self, x, flux, fwhm, temp, tau, scale, ratio):
        if self._perform_template_fitting:
            return evaluation.evaluate_template(
                x,
                flux, fwhm,
                template=self.template,
                interpolation_matrix=self._get_interpolation_matrix(x),
            )
        
        return evaluation.evaluate(
            x,
            flux, fwhm, temp, tau, scale, ratio,
            sigma_res=self.sigma_res,
            edge=self.edge,
            waves=self.waves,
            weights=self.weights,
            boltz=self.boltz,
            x_grid=self.x_grid,
            interpolation_matrix=self._get_interpolation_matrix(x),
        )

    def fit_deriv(self, x, flux, fwhm, temp, tau, scale, ratio):
        if self._perform_template_fitting:
            return evaluation.fit_deriv_template(
                x, 
                flux, fwhm, 
                template=self.template,
                interpolation_matrix=self._get_interpolation_matrix(x),
                fixed=self.fixed,
            )
        
        return evaluation.fit_deriv(
            x,
            flux, fwhm, temp, tau, scale, ratio,
            sigma_res=self.sigma_res,
            edge=self.edge,
            waves=self.waves,
            weights=self.weights,
            boltz=self.boltz,
            x_grid=self.x_grid,
            interpolation_matrix=self._get_interpolation_matrix(x),
            fixed=self.fixed,
        )
    
    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected BalmerModel instance, got {type(value)} instead."
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    @property
    def model_type(self) -> Literal['ba']:
        """
        Returns a string-representation of the model type.
        """
        return 'ba'
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (2.0, 0.0)
    
    def _get_interpolation_matrix(
        self,
        x_out: NDArray[float64],
    ) -> tuple[csr_matrix, NDArray[float64]]:
        """
        Retrieves an interpolation matrix from the cache which maps from the 
        BalmerTemplate's x-grid to the provided x_out grid. 
        """
        cache_key: tuple[int, int] = (x_out.size, hash(x_out.tobytes()))
        return self._interpolation_cache.get(cache_key, None)

    def _ignore_blue_side(self) -> None:
        """
        Ignore the blue side of the Balmer pseudo-continuum model by nullifying
        the (blackbody + attenuation) contribution. The attenuation is also 
        nullified.
        """
        self._prev_blue_fixed: dict[str, bool] = {
            'tau':  self.tau.fixed, 
            'scale': self.scale.fixed, 
            'ratio': self.ratio.fixed,
        }
        self.tau.fixed = True
        self.scale.fixed = True
        self.ratio.fixed = True

    def _restore_blue_side(self) -> None:
        """
        Restore the blue side of the Balmer pseudo-continuum model.
        """
        self.tau.fixed = self._prev_blue_fixed['tau']
        self.scale.fixed = self._prev_blue_fixed['scale']
        self.ratio.fixed = self._prev_blue_fixed['ratio']
        del self._prev_blue_fixed

    def _ignore_red_side(self) -> None:
        """
        Ignore the red side of the Balmer pseudo-continuum model by nullifying
        the (line series) contribution.
        """
        self._prev_red_fixed: dict[str, bool] = {
            'ratio': self.ratio.fixed,
        }
        self.ratio.fixed = True

    def _restore_red_side(self) -> None:
        """
        Restore the red side of the Balmer pseudo-continuum model.
        """
        self.ratio.fixed = self._prev_red_fixed['ratio']
        del self._prev_red_fixed

    @property
    def _blue_is_ignored(self) -> bool:
        return hasattr(self, '_prev_blue_fixed')

    @property
    def _red_is_ignored(self) -> bool:
        return hasattr(self, '_prev_red_fixed')

    @validate_call(validate_return=False)
    def rasterFit(
        self,
        x: FittableFloatVector,
        y: FittableFloatVector,
        dy: FittableFloatVector,
        *,
        raster_n: int = 20,
        inplace: bool = False,
    ) -> Self:
        """
        Performs a raster fit.

        Notes
        -----
        A raster fit is only supported for models with fixed 'ratio' parameters.
        The ratio parameter is held fixed during the raster fit, and its value
        is assigned to the best-fit model.
        """
        if not self.fwhm.fixed and (None in self.fwhm.bounds):
            msg = "Cannot perform raster fit with free 'fwhm' and \
                non-finite bounds."
            raise ValueError(msg)
        
        if not inplace: 
            return self.copy().rasterFit.__wrapped__(
                x, y, dy, raster_n=raster_n, inplace=True,
            )
        
        if self._perform_template_fitting:
            f = partial(
                evaluation.evaluate_template,
                x=x, 
                flux=1.0, 
                template=self.template_tuple,
                interpolation_matrix=self._get_interpolation_matrix(x),
            )
        else:
            f = partial(
                evaluation.evaluate,
                x=x, 
                flux=1.0, 
                temp=self.temp.value,
                tau=self.tau.value,
                scale=self.scale.value,
                ratio=self.ratio.value,
                sigma_res=self.sigma_res,
                edge=self.edge,
                waves=self.waves,
                weights=self.weights,
                boltz=self.boltz,
                x_grid=self.x_grid,
                interpolation_matrix=self._get_interpolation_matrix(x),
            )
        
        if self.fwhm.fixed:
            fwhms = array([self.fwhm.value], dtype=float)
            data = f(self.fwhm.value)[:,None]
        else:
            fwhms = (
                self.template.fwhm \
                if self._perform_template_fitting else 
                linspace(*self.fwhm.bounds, raster_n, dtype=float)
            )
            data = stack([f(_) for _ in fwhms], axis=0)

        # Chi2 loss function
        chi2s, fluxs = rasterise.__wrapped__(
            y, dy, fwhms, data,
            flux_bounds=self.flux.bounds,
            fwhm_bounds=self.fwhm.bounds,
        )

        if self.fwhm.fixed:
            self.flux.value = fluxs[0]
        else:
            idx: int = argmin(chi2s).flatten()[0]
            self.flux.value = fluxs[idx]
            self.fwhm.value = fwhms[idx]
            
        return self
    
    @validate_call(validate_return=False)
    def adjustFromPowerLaw(
        self,
        a_qsfit: float,
        model: PowerLawModel,
        info: Info,
        *,
        inplace: bool = False,
    ) -> Self:
        """
        From Calderone et al. (2017):

        Adjusts the Balmer model's flux based on the power law flux density
        at 3000 Å.

        Parameters
        ----------
        a_qsfit : float
            The flux density of the Balmer continuum relative to the power law 
            flux densityat 3000 Å. Calderone et al. (2017) use '0.1'.
        model : PowerLawModel
            The power law model used to estimate the flux density at 3000 Å.
        info : Info
            Instance of Info class used to convert 3000 Å to unitless 
            wavelengths.
        inplace: bool, optional
            If True, modifies this instance in-place. Otherwise, returns a copy.
            Default is False.
        """
        if not inplace:
            return self.copy().adjustFromPowerLaw.__wrapped__(
                a_qsfit, model, info, inplace=True,
        )
        wave: float = info.units.getWavelength(3000 * Unit('angstrom'))
        y_pl: float = model(wave)
        y_ba: float = self.evaluate(
            wave, 
            1.0, self.fwhm.value, self.temp.value, 
            self.tau.value, self.scale.value, self.ratio.value,
        )

        self.flux.value = clip(
            a_qsfit * y_pl / y_ba,
            lb if ((lb := self.flux.bounds[0]) is not None) else 0,
            ub if ((ub := self.flux.bounds[1]) is not None) else inf,
        )

        return self