"""
    Lorem ipsum.
"""

from typing import Self, Literal
from astropy.modeling import Parameter
from numpy import pi, dot, isclose
from scipy.stats import norm

from .utils import instantiate_model
from ..utils.basemodel import BaseModel
from ..utils.astropy import apply_bounds

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FittableFloatVector
from quasar_typing.bounds import AstropyBounds
from quasar_typing.misc.logger import Logger_

from . import evaluation

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

    Notes
    -----
    Lorem ipsum.
    """

    strength = Parameter(default=1, bounds=(0, None))
    sigma_v = Parameter(default=1e-3, bounds=(0, None))
    v_off = Parameter(default=0, bounds=(-1, 1))

    def __init__(
        self,
        wave: float,
        sigma_res: float,
        strength: float = 1.0,
        sigma_v: float = 1e-3,
        v_off: float = 0.0,
        n_sigmas: float = 3.0,
        **kwargs,
    ):
        super().__init__(
            strength, sigma_v, v_off,
            **kwargs,
        )
        self.wave: float = wave
        self.sigma_res: float = sigma_res
        self.n_sigmas: float = n_sigmas
    
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
            wave=self.wave, sigma_res=self.sigma_res, n_sigmas=self.n_sigmas,
        )
    
    def fit_deriv(self, x, strength, sigma_v, v_off):
        return evaluation.fit_deriv(
            x,
            strength, sigma_v, v_off,
            wave=self.wave, sigma_res=self.sigma_res, fixed=self.fixed,
        )
    
    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, GaussianModel):
            msg = "Expected GaussianModel, got {}".format(
                type(value).__name__,
            )
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    @validate_call(validate_return=False)
    @staticmethod
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
        """
        ** PYDANTIC VALIDATED FUNCTION **

        Lorem ipsum.

        Parameters
        ----------
        wave : float
        x : 1D numpy.array of floats
        y : 1D numpy.array of floats
        y_smooth : 1D numpy.array of floats
        name : str or None, optional
        strength_bounds : tuple of floats or None, optional
        v_off_bounds : tuple of floats or None, optional
        sigma_v_bounds : tuple of floats or None, optional
        sigma_res : float or None, optional
        logger : Logger_ or None, optional

        Returns
        -------
        GaussianModel

        Notes
        -----
        Lorem ipsum.
        """
        params = instantiate_model(
            wave, 
            x, 
            y, 
            y_smooth = y_smooth,
            sigma_res = sigma_res,
            strength_bounds = strength_bounds,
            v_off_bounds = v_off_bounds,
            sigma_v_bounds = sigma_v_bounds,
        )
        model = GaussianModel(wave, sigma_res, *params, name=name or 'model')
        model.strength.bounds = strength_bounds
        model.sigma_v .bounds = sigma_v_bounds
        model.v_off   .bounds = v_off_bounds

        if logger is not None:
            msg = "Instantiated GaussianModel with parameters: "
            msg += "('strength') {:.1e} < {:.1e} < {:.1e}, ".format(
                strength_bounds[0], params[0], strength_bounds[1],
            )
            msg += "('sigma_v') {:.1e} < {:.1e} < {:.1e}, ".format(
                sigma_v_bounds[0], params[1], sigma_v_bounds[1],
            )
            msg += "('v_off') {:.1e} < {:.1e} < {:.1e}.".format(
                v_off_bounds[0], params[2], v_off_bounds[1],
            )
            logger.debug(msg)
        
        return model
    
    ### Utility functions

    @property
    def pure_name(self) -> str:
        return self.name.split('#')[0]

    @property
    def mu(self) -> float:
        return self.wave * (1 + self.v_off.value)
    
    @property
    def sigma(self) -> float:
        return self.mu * (self.sigma_v.value**2 + self.sigma_res**2)**0.5

    @property
    def peak(self) -> float:
        return self.strength.value / (self.sigma * (2 * pi)**0.5)

    @property
    def model_type(self) -> Literal['em']:
        """
        Lorem ipsum.

        Returns
        -------
        Literal['em']

        Notes
        -----
        Lorem ipsum.
        """
        return 'em'
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Lorem ipsum.

        Returns
        -------
        tuple[float, float]

        Notes
        -----
        Lorem ipsum.
        """
        return (
            4.0,        # 4: line emission
            self.mu     # sort from bluest to reddest centre. 
        )
    
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
        profile = norm(self.mu, self.sigma)
        p = profile.pdf(x) * (x * self.sigma_res)
        
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
        dist = norm(self.mu, self.sigma)
        p = dist.pdf(x) * (x * self.sigma_res)
        
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
        dist = norm(self.mu, self.sigma)
        p = dist.pdf(x) * (x * self.sigma_res)
        
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
        dist = norm(self.mu, self.sigma)
        p = dist.pdf(x) * (x * self.sigma_res)
        return dot(p, is_absorbed)
    
    def copy(self) -> Self:
        """
        Lorem ipsum.

        Returns
        -------
        GaussianModel

        Notes
        -----
        Lorem ipsum.
        """
        new = super().copy()
        new.wave = self.wave
        new.sigma_res = self.sigma_res
        new.n_sigmas = self.n_sigmas
        return new
    
    @validate_call(validate_return=False)
    def makeCopy(
        self, 
        x: FittableFloatVector, 
        dy: FittableFloatVector,
        z: FittableFloatVector,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **

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
        """
        Lorem ipsum.

        Returns
        -------
        bool

        Notes
        -----
        Lorem ipsum.
        """
        if self.has_bounds:
            for attr in ['strength', 'sigma_v', 'v_off']:
                param = getattr(self, attr)

                # Exception: if the parameter is fixed, we ignore it even if 
                # it's touching bounds.
                if param.fixed: continue

                val: float = param.value
                lb: float | None = param.bounds[0]
                ub: float | None = param.bounds[1]

                if lb is not None and isclose(val, lb): return True
                if ub is not None and isclose(val, ub): return True

        return False