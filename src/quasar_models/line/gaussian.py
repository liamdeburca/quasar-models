"""
    Lorem ipsum.
"""
from typing import Self, Literal, ClassVar
from astropy.modeling import Parameter
from math import hypot
from numpy import pi, dot, isclose
from scipy.stats import norm

from quasar_models.utils.basemodel import BaseModel
from quasar_models.utils.astropy import apply_bounds

from quasar_typing.numpy import FittableFloatVector, FloatVector
from quasar_typing.bounds import AstropyBounds
from quasar_typing.logging import Logger_

from quasar_utils.decorators import validate_call

from . import evaluation
from .utils import instantiate_model

from quasar_models.tying.base_parameter_tie import BaseParameterTie

N_SIGMAS:  float = 3.0
GAUSS_AMP: float = 1 / (2 * pi)**0.5

class GaussianModel(BaseModel):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength : Parameter
        Line strength (flux integral). Defaults to 1 with positive bounds.
    sigma_v : Parameter
        Velocity dispersion in units of $c$. Defaults to 1e-3 with positive 
        bounds.
    v_off : Parameter
        Velocity offset relative to rest wavelength. Defaults to 0 with bounds
        [-1, 1].
    wave : float
        Rest wavelength of the emission line.
    sigma_res : float
        Velocity resolution of the spectrum in units of $c$.
    n_sigmas : float
        Number of Gaussian sigmas to include in sparse evaluation. Defaults to 
        3.0.
    """
    strength = Parameter(
        default=1.0, 
        bounds=(0.0, None),
        fixed=False,
    )
    sigma_v = Parameter(
        default=1e-3, 
        bounds=(0.0, None),
        fixed=False,
    )
    v_off = Parameter(
        default=0.0, 
        bounds=(-1.0, 1.0),
        fixed=False,
    )

    model_type: ClassVar[Literal['em']] = 'em'

    @classmethod
    def create(
        cls,
        wave: float,
        sigma_res: float,
        *,
        strength: float = 1.0,
        sigma_v: float = 1e-3,
        v_off: float = 0.0,
        n_sigmas: float = 3.0,
        **kwargs,
    ) -> Self:
        return GaussianModel(
            strength, sigma_v, v_off,
            meta={'wave': wave, 'sigma_res': sigma_res, 'n_sigmas': n_sigmas},
            **kwargs,
        )
    
    @property
    def wave(self) -> float: 
        return self.meta['wave']

    @wave.setter
    def wave(self, value: float) -> None:
        self.meta['wave'] = value

    @property
    def sigma_res(self) -> float: 
        return self.meta['sigma_res']

    @sigma_res.setter
    def sigma_res(self, value: float) -> None:
        self.meta['sigma_res'] = value

    @property
    def n_sigmas(self) -> float: 
        return self.meta['n_sigmas']

    @n_sigmas.setter
    def n_sigmas(self, value: float) -> None:
        self.meta['n_sigmas'] = value
    
    def evaluate(self, x, strength, sigma_v, v_off):
        return evaluation.evaluate(
            x, 
            strength, sigma_v, v_off, 
            wave=self.wave, sigma_res=self.sigma_res,
        )
    
    def evaluate_sparse(self, x, strength, sigma_v, v_off):
        return evaluation.evaluate_sparse(
            x,
            strength, sigma_v, v_off,
            wave=self.wave, 
            sigma_res=self.sigma_res, 
            n_sigmas=self.n_sigmas,
        )
    
    def fit_deriv(self, x, strength, sigma_v, v_off):
        return evaluation.fit_deriv(
            x,
            strength, sigma_v, v_off,
            wave=self.wave, 
            sigma_res=self.sigma_res, 
            fixed=self.fixed_dict,
        )
    
    @staticmethod
    @validate_call
    def instantiate(
        wave: float,
        x: FittableFloatVector,
        y: FittableFloatVector,
        y_smooth: FittableFloatVector,
        *,
        name: str | None = None,
        strength_bounds: AstropyBounds | None = None,
        v_off_bounds: AstropyBounds | None = None,
        sigma_v_bounds: AstropyBounds | None = None,
        sigma_res: float | None = None,
        logger: Logger_ | None = None,
    ) -> Self:
        strength, sigma_v, v_off = instantiate_model(
            wave, 
            x, 
            y, 
            y_smooth=y_smooth,
            sigma_res=sigma_res,
            strength_bounds=strength_bounds,
            v_off_bounds=v_off_bounds,
            sigma_v_bounds=sigma_v_bounds,
        )
        model = GaussianModel.create(
            wave, 
            sigma_res, 
            strength=strength, 
            sigma_v=sigma_v,
            v_off=v_off,
            name=name or 'model',
        )
        model.strength.bounds = strength_bounds
        model.sigma_v .bounds = sigma_v_bounds
        model.v_off   .bounds = v_off_bounds

        if logger is not None:
            msg = "Instantiated GaussianModel with parameters: "
            msg += "('strength') {:.1e} < {:.1e} < {:.1e}, ".format(
                strength_bounds[0], strength, strength_bounds[1],
            )
            msg += "('sigma_v') {:.1e} < {:.1e} < {:.1e}, ".format(
                sigma_v_bounds[0], sigma_v, sigma_v_bounds[1],
            )
            msg += "('v_off') {:.1e} < {:.1e} < {:.1e}.".format(
                v_off_bounds[0], v_off, v_off_bounds[1],
            )
            logger.debug(msg)
        
        return model
    
    ### Utility functions

    @property
    def mu(self) -> float:
        return self.wave * (1 + self.v_off.value)
    
    @property
    def sigma(self) -> float:
        return self.mu * hypot(self.sigma_v.value, self.sigma_res)

    @property
    def peak(self) -> float:
        return self.strength.value / (self.sigma * (2 * pi)**0.5)
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        return (4.0, self.mu)
    
    ###

    def getPeakSNR(self, obj: object) -> float:
        """
        Lorem ipsum.

        Parameters
        ----------
        obj : object

        Returns
        -------
        float

        Notes
        -----
        Lorem ipsum.
        """
        x, _, dy, _ = obj.getMaskedCoords(without_absorption=True)
        p = norm.pdf(x, self.mu, self.sigma) * (x * self.sigma_res)        
        return self.peak / dot(p, dy)
    
    def getFluxSNR(self, obj: object) -> float:
        """
        Lorem ipsum.

        Parameters
        ----------
        obj : object

        Returns
        -------
        float

        Notes
        -----
        Lorem ipsum.
        """
        x, y, dy, _ = obj.getMaskedCoords(without_absorption=True)
        p = norm.pdf(x, self.mu, self.sigma) * (x * self.sigma_res)        
        return dot(p, y) / dot(p, dy)
    
    def getLineSNR(self, obj: object) -> float:
        """
        Lorem ipsum.

        Parameters
        ----------
        obj : object

        Returns
        -------
        float

        Notes
        -----
        Lorem ipsum.
        """
        x, _, dy, _ = obj.getMaskedCoords(without_absorption=True)
        p = norm.pdf(x, self.mu, self.sigma) * (x * self.sigma_res)        
        return self.strength.value / dot(p, dy)
    
    def getWeightedAbsorption(self, obj: object) -> float:
        """
        Lorem ipsum.

        Parameters
        ----------
        obj : object

        Returns
        -------
        float

        Notes
        -----
        Lorem ipsum.
        """        
        x, _, _, _, is_absorbed = obj.getMaskedCoords(without_absorption=False)
        p = norm.pdf(x, self.mu, self.sigma) * (x * self.sigma_res)
        return dot(p, is_absorbed)
    
    @validate_call
    def makeCopy(
        self, 
        x: FittableFloatVector, 
        dy: FittableFloatVector,
        z: FittableFloatVector,
    ) -> Self:
        """
        Lorem ipsum.

        Parameters
        ----------
        x : 1D numpy.array of floats
        dy : 1D numpy.array of floats
        z : 1D numpy.array of floats

        Returns
        -------
        GaussianModel

        Notes
        -----
        Lorem ipsum.
        """        
        new = self.copy()
        if not (new.strength.fixed or new.strength.tied):
            new.strength.value = apply_bounds(
                dot(z * dy, x * new.sigma_res),
                new.strength.bounds
            )

        if not (new.sigma_v.fixed or new.sigma_v.tied):
            new.sigma_v.value = apply_bounds(
                0.5 * (new.sigma_v.value + new.sigma_v.bounds[0]),
                new.sigma_v.bounds
            )

        if not (new.v_off.fixed or new.v_off.tied):
            new.v_off.value = apply_bounds(
                0.5 * new.v_off.value,
                new.v_off.bounds
            )

        return new
    
    def isTouchingBounds(self) -> bool:
        if self.has_bounds:
            for attr in ['strength', 'sigma_v', 'v_off']:
                param = getattr(self, attr)

                if param.fixed: 
                    continue

                val: float = param.value
                lb: float | None = param.bounds[0]
                ub: float | None = param.bounds[1]

                if lb is not None and isclose(val, lb): 
                    return True
                if ub is not None and isclose(val, ub): 
                    return True

        return False