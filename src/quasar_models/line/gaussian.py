"""
Lorem ipsum.
"""

from math import hypot, log, pi
from typing import Any, ClassVar, Literal, Self

from astropy.constants import c
from astropy.modeling import Parameter
from numpy import dot
from quasar_typing.bounds import AstropyBounds
from quasar_typing.logging import Logger_
from quasar_typing.numpy import FittableFloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info
from scipy.stats import norm

from quasar_models._core.modeling.gaussian import (
    GaussianEvaluateV,
    GaussianEvaluateX,
    GaussianFitDerivV,
    GaussianFitDerivX,
    choose_evaluate_func,
    choose_fit_deriv_func,
    evaluate_v,
    fit_deriv_v_all,
)
from quasar_models.modeling import BaseModel

from ..utils.astropy import apply_bounds
from ..utils.serialization import (
    deserialize_parameter,
    deserialize_quantity,
    serialize_parameter,
    serialize_quantity,
)
from .utils import instantiate_model

N_SIGMAS: float = 3.0
GAUSS_AMP: float = 1 / (2 * pi) ** 0.5
SIGMA_TO_FWHM: float = 2 * (2 * log(2)) ** 0.5
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM

C_KMS: float = c.to("km/s").value  # Exact speed of light in km/s
INV_C_KMS: float = 1.0 / C_KMS


class GaussianModel(BaseModel):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength : Parameter
        Line strength (flux integral). Defaults to 1 with positive bounds.
    fwhm_v : Parameter
        Full width at half maximum in velocity units. Defaults to 1e-3 with positive
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
        description="Integrated flux",
        default=1.0,
        bounds=(0.0, None),
        fixed=False,
    )
    fwhm_v = Parameter(
        description="Intrinsic FWHM (km/s)",
        default=1000,
        bounds=(0.0, None),
        fixed=False,
    )
    v_off = Parameter(
        description="Velocity offset (km/s) from the theoretical centre",
        default=0.0,
        bounds=(-C_KMS, C_KMS),
        fixed=False,
    )

    model_type: ClassVar[Literal["em"]] = "em"

    def __str__(self) -> str:
        name = self.name
        strength = self.strength.value
        fwhm_v = self.fwhm_v.value
        v_off = self.v_off.value
        return f"{self.__class__.__name__}"\
            f"({name=}, {strength=:.1f}, {fwhm_v=:.1f}, {v_off=:.1f})"

    def __repr__(self) -> str:
        return self.__str__()
    
    @classmethod
    def create(
        cls,
        wave: float,
        sigma_res: float,
        linetype: Literal["n", "b"],
        *,
        strength: float = 1.0,
        fwhm_v: float = 1000.0,
        v_off: float = 0.0,
        n_sigmas: float = 3.0,

        strength_bounds: AstropyBounds | None = None,
        fwhm_v_bounds: AstropyBounds | None = None,
        v_off_bounds: AstropyBounds | None = None,

        strength_fixed: bool | None = None,
        fwhm_v_fixed: bool | None = None,
        v_off_fixed: bool | None = None,
        **kwargs,
    ) -> Self:
        """
        Parameters
        ----------
        wave : float
            Theoretical rest wavelength of the emission line.
        sigma_res : float
            Velocity resolution (c) of the spectrum.
        strength : float, optional
        fwhm_v : float, optional
        v_off : float, optional
        strength_bounds : tuple[float, float] | None, optional
        fwhm_v_bounds : tuple[float, float] | None, optional
        v_off_bounds : tuple[float, float] | None, optional
        n_sigmas : float, optional
        """
        model = GaussianModel(
            strength,
            fwhm_v,
            v_off,
            meta={
                "wave": wave, 
                "sigma_res": sigma_res, 
                "n_sigmas": n_sigmas,
                "linetype": linetype,
            },
            **kwargs,
        )
        if strength_bounds is not None:
            model.strength.value = apply_bounds.__wrapped__(
                strength, 
                strength_bounds,
            )
            model.strength.bounds = strength_bounds
        if fwhm_v_bounds is not None:
            model.fwhm_v.value = apply_bounds.__wrapped__(
                fwhm_v, 
                fwhm_v_bounds,
            )
            model.fwhm_v.bounds = fwhm_v_bounds
        if v_off_bounds is not None:
            model.v_off.value = apply_bounds.__wrapped__(
                v_off,
                v_off_bounds,
            )
            model.v_off.bounds = v_off_bounds

        if strength_fixed is not None:
            model.strength.fixed = strength_fixed
        if fwhm_v_fixed is not None:
            model.fwhm_v.fixed = fwhm_v_fixed
        if v_off_fixed is not None:
            model.v_off.fixed = v_off_fixed

        return model

    @property
    def wave(self) -> float:
        """
        Theoretical rest wavelength of the emission line.
        """
        return self.meta["wave"]

    @wave.setter
    def wave(self, value: float) -> None:
        self.meta["wave"] = value

    @property
    def sigma_res(self) -> float:
        """
        Velocity resolution of the spectrum (c).
        """
        return self.meta["sigma_res"]

    @sigma_res.setter
    def sigma_res(self, value: float) -> None:
        self.meta["sigma_res"] = value

    @property
    def linetype(self) -> Literal["n", "b"]:
        return self.meta["linetype"]

    @linetype.setter
    def linetype(self, value: Literal["n", "b"]) -> None:
        self.meta["linetype"] = value

    @property
    def v_res(self) -> float:
        """
        Velocity resolution of the spectrum (km/s).
        """
        return self.sigma_res * C_KMS

    @property
    def dx(self) -> float | None:
        """
        Wavelength resolution of the spectrum (near the rest wavelength).
        """
        return self.meta.get("dx", None)

    @dx.setter
    def dx(self, value: float) -> None:
        self.meta["dx"] = value

    @dx.deleter
    def dx(self) -> None:
        self.meta.pop("dx", None)

    @property
    def n_sigmas(self) -> float:
        return self.meta["n_sigmas"]

    @n_sigmas.setter
    def n_sigmas(self, value: float) -> None:
        self.meta["n_sigmas"] = value

    @property
    def kwargs(self) -> dict[str, float]:
        return self.meta.get(
            "kwargs", 
            {"wave": self.wave, "sigma_res": self.sigma_res},
        )

    @kwargs.setter
    def kwargs(self, value: dict[str, float]) -> None:
        self.meta["kwargs"] = value

    @kwargs.deleter
    def kwargs(self) -> None:
        self.meta.pop("kwargs", None)

    def _set_kwargs(self) -> None:
        if self.dx is None:
            self.kwargs = {"wave": self.wave, "sigma_res": self.sigma_res}
        else:
            self.kwargs = {"wave": self.wave, "dx": self.dx}
        
    def evaluate(self, x, strength, fwhm_v, v_off, y=None):            
        return self.evaluate_func(
            x,
            *self._transform_args_if_ndarray(strength, fwhm_v, v_off),
            **self.kwargs,
            y=y,
        )

    def partial_deriv(self, x, strength, fwhm_v, v_off, derivs=None):
        return self.fit_deriv_func(
            x,
            *self._transform_args_if_ndarray(strength, fwhm_v, v_off),
            **self.kwargs,
            derivs=derivs,
        )

    def fit_deriv(self, x, strength, fwhm_v, v_off, derivs=None):
        return list(self.partial_deriv(x, strength, fwhm_v, v_off, derivs=derivs))

    ### Model preparation

    @property
    def evaluate_func(self) -> GaussianEvaluateV | GaussianEvaluateX:
        return self.meta.get("evaluate_func", evaluate_v)

    @evaluate_func.setter
    def evaluate_func(self, value: GaussianEvaluateV | GaussianEvaluateX) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> GaussianFitDerivV | GaussianFitDerivX:
        return self.meta.get("fit_deriv_func", fit_deriv_v_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: GaussianFitDerivV | GaussianFitDerivX) -> None:
        self.meta["fit_deriv_func"] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop("fit_deriv_func", None)

    def _choose_evaluate_func(self) -> None:
        self.evaluate_func = choose_evaluate_func(self.dx is None)

    def _choose_fit_deriv_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.dx is None, 
            fixed or self.fixed,
        )

    def prepare_model(
        self, 
        dx: float | None,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        if dx is None:
            del self.dx
        else:
            self.dx = dx
        super().prepare_model(fixed=fixed)

    def unprepare_model(self) -> None:
        del self.dx
        super().unprepare_model()

    @classmethod
    @validate_call
    def instantiate(
        cls,
        wave: float,
        x: FittableFloatVector,
        y: FittableFloatVector,
        y_smooth: FittableFloatVector,
        *,
        sigma_res: float,
        linetype: str,
        name: str | None = None,
        strength_bounds: AstropyBounds | None = None,
        v_off_bounds: AstropyBounds | None = None,
        fwhm_v_bounds: AstropyBounds | None = None,
        strength_fixed: bool | None = None,
        v_off_fixed: bool | None = None,
        fwhm_v_fixed: bool | None = None,
        logger: Logger_ | None = None,
    ) -> Self:
        strength, fwhm_v, v_off = instantiate_model(
            wave,
            x,
            y,
            y_smooth=y_smooth,
            sigma_res=sigma_res,
            strength_bounds=strength_bounds,
            v_off_bounds=v_off_bounds,
            fwhm_v_bounds=fwhm_v_bounds,
            linetype=linetype,
        )
        model = GaussianModel.create(
            wave,
            sigma_res,
            linetype,
            strength=strength,
            fwhm_v=fwhm_v,
            v_off=v_off,
            strength_bounds=strength_bounds,
            fwhm_v_bounds=fwhm_v_bounds,
            v_off_bounds=v_off_bounds,
            strength_fixed=strength_fixed,
            fwhm_v_fixed=fwhm_v_fixed,
            v_off_fixed=v_off_fixed,
            name=name or "model",
        )
        if logger is not None:
            msg = "Instantiated GaussianModel with parameters: "
            for p in (getattr(model, pname) for pname in model.param_names):
                val = p.value
                lb, ub = p.bounds
                msg += f"({p.name}) {lb:.1f} < {val:.1f} < {ub:.1f}, "
            logger.debug(msg.removesuffix(", "))

        return model

    ### Utility functions

    @property
    def is_narrow(self) -> bool:
        return self.linetype == "n"

    @property
    def is_broad(self) -> bool:
        return not self.is_narrow

    @property
    def mu(self) -> float:
        return self.wave * (1 + self.v_off.value * INV_C_KMS)

    @property
    def sigma_v(self) -> float:
        return self.fwhm_v.value * FWHM_TO_SIGMA

    @property
    def sigma(self) -> float:
        return self.mu * hypot(self.sigma_v * INV_C_KMS, self.sigma_res)

    @property
    def fwhm(self) -> float:
        return self.sigma * SIGMA_TO_FWHM

    @property
    def peak(self) -> float:
        return self.strength.value / (self.sigma * (2 * pi) ** 0.5)

    @property
    def sorting_key(self) -> tuple[float, float, int]:
        return (4.0, self.mu, 0 if self.linetype == "n" else 1)

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
                new.strength.bounds,
            )

        if not (new.fwhm_v.fixed or new.fwhm_v.tied):
            new.fwhm_v.value = apply_bounds(
                0.5 * (new.fwhm_v.value + new.fwhm_v.bounds[0]), 
                new.fwhm_v.bounds,
            )

        if not (new.v_off.fixed or new.v_off.tied):
            new.v_off.value = apply_bounds(
                0.5 * new.v_off.value, 
                new.v_off.bounds,
            )

        return new

    def isTouchingBounds(self) -> bool:
        if self.has_bounds:
            for attr in ["strength", "fwhm_v", "v_off"]:
                param = getattr(self, attr)

                if param.fixed:
                    continue

                val: float = param.value
                lb: float | None = param.bounds[0]
                ub: float | None = param.bounds[1]

                if lb is not None and val <= lb:
                    return True
                if ub is not None and ub <= val:
                    return True

        return False

    ### Serialization

    def serialize(
        self,
        info: Info,
    ) -> dict[str, dict[str, Any]]:
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        data = {
            "wave": serialize_quantity(self.wave, wave_unit),
            "linetype": self.linetype,
            "n_sigmas": self.n_sigmas,
            "strength": serialize_parameter(self.strength, strength_unit),
            "fwhm_v": serialize_parameter(self.fwhm_v, None),
            "v_off": serialize_parameter(self.v_off, None),
        }
        return {f"GaussianModel::{self.name}": data}

    @classmethod
    def deserialize(
        cls,
        data: dict[str, Any],
        name: str,
        info: Info,
    ) -> Self:
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        wave = deserialize_quantity(data["wave"], wave_unit)
        linetype = data["linetype"]
        n_sigmas = data["n_sigmas"]
        strength = deserialize_parameter(data["strength"], strength_unit)
        fwhm_v = deserialize_parameter(data["fwhm_v"], None)
        v_off = deserialize_parameter(data["v_off"], None)

        out = GaussianModel.create(
            wave, info.loading.sigma_res, linetype,
            name=name,
            strength=strength["value"],
            fwhm_v=fwhm_v["value"],
            v_off=v_off["value"],
            n_sigmas=n_sigmas,
            strength_bounds=strength["bounds"],
            fwhm_v_bounds=fwhm_v["bounds"],
            v_off_bounds=v_off["bounds"],
            strength_fixed=strength["fixed"],
            fwhm_v_fixed=fwhm_v["fixed"],
            v_off_fixed=v_off["fixed"],
        )
        out.strength.tied = strength["tied"]
        out.fwhm_v.tied = fwhm_v["tied"]
        out.v_off.tied = v_off["tied"]

        return out