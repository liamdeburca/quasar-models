__all__ = ["_VProfileCopy"]

from collections.abc import Iterable
from itertools import product
from typing import Any, ClassVar, Literal, Self, TypedDict, Union

from astropy.constants import c
from numpy import float64, fromiter
from quasar_typing.astropy import CompoundModel_
from quasar_typing.bounds import AstropyBounds
from quasar_utils.setup import Info

from quasar_models._core.modeling.vprofilecopy import (
    VProfileCopyEvaluate,
    VProfileCopyFitDeriv,
    choose_evaluate_func,
    choose_fit_deriv_func,
    evaluate_v,
    fit_deriv_v_all,
)
from quasar_models.line.gaussian import GaussianModel
from quasar_models.modeling import BaseModel, LinearTie

from ...utils.astropy import apply_bounds
from ...utils.serialization import (
    deserialize_parameter,
    deserialize_quantity,
    serialize_parameter,
    serialize_quantity,
)

C_KMS: float = c.to("km/s").value  # Exact speed of light in km/s


class MetaData(TypedDict):
    wave: float
    sigma_res: float
    linetype: Literal["n", "b"]
    dx: float | None
    kwargs: dict[str, float]
    master_name: str
    n_sigmas: float


class _VProfileCopy(BaseModel):
    n_profiles: ClassVar[Literal[1, 2, 3, 4, 5]]
    model_type: ClassVar[Literal["em"]] = "em"

    @property
    def pure_name(self) -> str:
        return self.name

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
    def sigma_res(self) -> float | None:
        """
        Velocity resolution of the spectrum (c).
        """
        return self.meta["sigma_res"]

    @sigma_res.setter
    def sigma_res(self, value: float | None) -> None:
        self.meta["sigma_res"] = value

    @property
    def linetype(self) -> Literal["n", "b"]:
        return self.meta["linetype"]

    @linetype.setter
    def linetype(self, value: Literal["n", "b"]) -> None:
        self.meta["linetype"] = value

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

    @property
    def v_res(self) -> float | None:
        """
        Velocity resolution of the spectrum (km/s).
        """
        if self.sigma_res is None:
            return None
        return self.sigma_res * C_KMS

    @property
    def master_name(self) -> str | None:
        return self.meta["master_name"]

    @master_name.setter
    def master_name(self, value: str | None) -> None:
        self.meta["master_name"] = value

    @property
    def n_sigmas(self) -> float | None:
        return self.meta["n_sigmas"]

    @n_sigmas.setter
    def n_sigmas(self, value: float | None) -> None:
        self.meta["n_sigmas"] = value

    @classmethod
    def create(
        cls,
        wave: float,
        name: str,
        linetype: Literal["n", "b"],
        *gs: GaussianModel,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: AstropyBounds | None = None,
        strength_scale_fixed: bool | None = None,
        sigma_res: float | None = None,
        master_name: str | None = None,
        n_sigmas: float | None = None,
        dx: float | None = None,
        adapt: bool = True,
    ) -> Self:
        if isinstance(gs, tuple) and len(gs) != cls.n_profiles:
            raise ValueError(
                "No. of Gaussian models does not match the number of profiles: "
                f"n_profiles = {cls.n_profiles}, n_gaussians = {len(gs)}"
            )

        meta = cls._get_metadata(
            wave,
            name,
            linetype,
            *gs,
            sigma_res=sigma_res,
            master_name=master_name,
            n_sigmas=n_sigmas,
            dx=dx,
        )
        model = cls(
            strength_scale_value,
            name=name,
            meta=meta,
        )
        if strength_scale_bounds is not None:
            model.strength_scale.value = apply_bounds.__wrapped__(
                strength_scale_value, strength_scale_bounds
            )
            model.strength_scale.bounds = strength_scale_bounds

        if strength_scale_fixed is not None:
            model.strength_scale.fixed = strength_scale_fixed

        if adapt:
            model._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)

        return model

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        linetype: Literal["n", "b"],
        model: Union[GaussianModel, Iterable[GaussianModel]],
        *,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,
        adapt: bool = True,
    ) -> Self:
        if cls.n_profiles == 1:
            assert isinstance(model, GaussianModel)
            model = (model,)
        if cls.n_profiles > 1:
            model = tuple(model)
            assert len(model) == cls.n_profiles

        model = cls.create(
            wave,
            name,
            linetype,
            *model,
            strength_scale_value=strength_scale_value,
            strength_scale_bounds=strength_scale_bounds,
            strength_scale_fixed=strength_scale_fixed,
            adapt=adapt,
        )
        return model

    def makeCopy(
        self,
        master_model: Union[GaussianModel, Iterable[GaussianModel]],
        *,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,
    ) -> Self:
        return self.from_model(
            self.wave,
            self.name,
            self.linetype,
            master_model,
            strength_scale_value=strength_scale_value,
            strength_scale_bounds=strength_scale_bounds,
            strength_scale_fixed=strength_scale_fixed,
            adapt=True,
        )

    @classmethod
    def _get_metadata(
        cls,
        wave: float,
        name: str,
        linetype: Literal["n", "b"],
        *gs: GaussianModel,
        sigma_res: float | None = None,
        master_name: str | None = None,
        n_sigmas: float | None = None,
        dx: float | None = None,
    ) -> MetaData:
        meta = {
            "pure_name": name,
            "wave": wave,
            "linetype": linetype,
        }

        if len(gs) == 0:
            # Used when copying
            assert sigma_res is not None
            assert master_name is not None
            assert n_sigmas is not None

            meta["sigma_res"] = sigma_res
            meta["master_name"] = master_name
            meta["n_sigmas"] = n_sigmas
            meta["dx"] = dx # Is allowed to be None

        elif len(gs) == cls.n_profiles:
            meta["sigma_res"] = gs[0].sigma_res
            meta["master_name"] = gs[0].pure_name
            meta["n_sigmas"] = gs[0].n_sigmas
            meta["dx"] = gs[0].dx

        return meta

    def __str__(self) -> str:
        name = self.name
        master = self.master_name
        wave = self.wave
        scale = self.strength_scale.value
        s = f"{self.__class__.__name__}({name=}, {master=}, "\
            f"{wave=:.2f}, {scale=:.1f})"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def _get_params_from_args(cls, n_profiles: int, *args) -> tuple[float, ...]:
        args = cls._transform_args_if_ndarray(*args)
        return tuple(
            fromiter((args[3 * i + o] for i in range(n_profiles)), dtype=float64)
            for o in range(3)
        )

    def evaluate(self, x, strength_scale, *args, y=None):
        strength_scale = self._transform_if_ndarray(strength_scale)
        params = self._get_params_from_args(self.n_profiles, *args)
        return self.evaluate_func(
            x,
            strength_scale, *params,
            **self.kwargs,
            y=y,
        )

    def partial_deriv(self, x, strength_scale, *args, derivs=None):
        strength_scale = self._transform_if_ndarray(strength_scale)
        params = self._get_params_from_args(self.n_profiles, *args)
        return self.fit_deriv_func(
            x,
            strength_scale, *params,
            **self.kwargs,
            derivs=derivs,
        )

    def fit_deriv(self, x, strength_scale, *args, derivs=None):
        return list(self.partial_deriv(x, strength_scale, *args, derivs=derivs))

    ### Model preparation

    @property
    def evaluate_func(self) -> VProfileCopyEvaluate:
        return self.meta.get("evaluate_func", evaluate_v)

    @evaluate_func.setter
    def evaluate_func(self, value: VProfileCopyEvaluate) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> VProfileCopyFitDeriv:
        return self.meta.get("fit_deriv_func", fit_deriv_v_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: VProfileCopyFitDeriv) -> None:
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
        dx: float | None = None,
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

    ###

    @property
    def sorting_key(self) -> tuple[float, float, int]:
        return (4.0, self.wave, 0 if self.linetype == "n" else 1)

    ### Utilities

    def _adapt_to_models(
        self,
        *gs: GaussianModel,
        tie_vel_profile: bool = True,
        inplace: bool = False,
    ) -> Self:
        """
        Lorem ipsum.

        Parameters
        ----------
        *gs : GaussianModel
        tie_vel_profile : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        if len(gs) != self.n_profiles:
            raise ValueError(
                "No. of Gaussian models does not match the number of profiles: "
                f"n_profiles={self.n_profiles} != {len(gs)}"
            )
        
        model = self if inplace else self.copy()

        for (i, g), pname in product(
            enumerate(gs, start=1),
            ("strength", "fwhm_v", "v_off"),
        ):
            attr_name: str = f"{pname}_{i}"
            attr = getattr(model, attr_name)
            attr.value = getattr(g, pname).value
            attr.bounds = getattr(g, pname).bounds

            if tie_vel_profile:
                attr.tied = LinearTie(
                    a=1.0,
                    b=0.0,
                    model_name=g.name,
                    parameter_name=pname,
                )

        return model

    def adaptStrengthScale(
        self,
        model: CompoundModel_[GaussianModel] | Iterable[GaussianModel],
    ) -> None:
        """
        Lorem ipsum.

        Parameters
        ----------
        model : CompoundModel_[GaussianModel] or Iterable[GaussianModel]

        Notes
        -----
        Lorem ipsum.
        """
        current_strength = sum(
            getattr(self, f"strength_{i}").value for i in range(1, self.n_profiles + 1)
        )
        model_strength = sum(
            m.strength.value
            for m in (model if isinstance(model, Iterable) else [model])
        )
        self.strength_scale.value = apply_bounds(
            model_strength / current_strength,
            self.strength_scale.bounds,
        )

    def splitIntoGaussians(self) -> list[GaussianModel]:
        """
        Lorem ipsum.

        Returns
        -------
        list[GaussianModel]

        Notes
        -----
        Lorem ipsum.
        """
        gaussian_models: list[GaussianModel] = []

        names: list[str] = (
            [self.pure_name]
            if self.n_profiles == 1
            else [self.pure_name + f"#{1 + i}" for i in range(self.n_profiles)]
        )
        for i, name in enumerate(names):
            g = GaussianModel.create(
                self.wave, 
                self.sigma_res, 
                self.linetype,
                name=name,
            )
            for attr_name in ("strength", "fwhm_v", "v_off"):
                getattr(g, attr_name).value = getattr(
                    self, f"{attr_name}_{i + 1}"
                ).value
                getattr(g, attr_name).bounds = getattr(
                    self, f"{attr_name}_{i + 1}"
                ).bounds

            gaussian_models.append(g)

        return gaussian_models

    ### Serialization

    def serialize(self, info: Info) -> dict[str, dict[str, Any]]:
        # Follow GaussianModel's serialization scheme but handle multiple
        # profile parameters based on cls.n_profiles.
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)
        kms_unit = "km/s"

        data: dict[str, Any] = {
            "wave": serialize_quantity(self.wave, wave_unit),
            "linetype": self.linetype,
            "n_sigmas": self.n_sigmas,
            # Meta fields necessary to reconstruct the object without
            # reference Gaussians
            "sigma_res": self.sigma_res,
            "master_name": self.master_name,
            "dx": self.dx,
            "strength_scale": serialize_parameter(self.strength_scale, None),
        }

        # Per-profile parameters
        for i in range(1, self.n_profiles + 1):
            data[f"strength_{i}"] = serialize_parameter(
                getattr(self, f"strength_{i}"), strength_unit
            )
            data[f"fwhm_v_{i}"] = serialize_parameter(
                getattr(self, f"fwhm_v_{i}"), None
            )
            data[f"v_off_{i}"] = serialize_parameter(
                getattr(self, f"v_off_{i}"), None
            )

        return {f"{self.__class__.__name__}::{self.name}": data}

    @classmethod
    def deserialize(cls, data: dict[str, Any], name: str, info: Info) -> Self:
        # Expect the inner data dict (not the outer wrapper)
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        wave = deserialize_quantity(data["wave"], wave_unit)
        linetype = data["linetype"]
        n_sigmas = data.get("n_sigmas")

        sigma_res = data.get("sigma_res")
        master_name = data.get("master_name")
        dx = data.get("dx")

        strength_scale_ser = deserialize_parameter(data["strength_scale"], None)

        # Build metadata required by create when no source Gaussians are
        # available (len(gs) == 0 branch).
        meta = cls._get_metadata(
            wave,
            name,
            linetype,
            sigma_res=sigma_res,
            master_name=master_name,
            n_sigmas=n_sigmas,
            dx=dx,
        )

        # Instantiate the object and populate parameters
        model = cls(
            strength_scale_ser["value"],
            name=name,
            meta=meta,
        )

        # Restore bounds/fixed/tied for strength_scale
        model.strength_scale.bounds = strength_scale_ser["bounds"]
        model.strength_scale.fixed = strength_scale_ser["fixed"]
        model.strength_scale.tied = strength_scale_ser["tied"]

        # Per-profile parameters
        for i in range(1, cls.n_profiles + 1):
            s = deserialize_parameter(data[f"strength_{i}"], strength_unit)
            f = deserialize_parameter(data[f"fwhm_v_{i}"], None)
            v = deserialize_parameter(data[f"v_off_{i}"], None)

            getattr(model, f"strength_{i}").value = s["value"]
            getattr(model, f"strength_{i}").bounds = s["bounds"]
            getattr(model, f"strength_{i}").fixed = s["fixed"]
            getattr(model, f"strength_{i}").tied = s["tied"]

            getattr(model, f"fwhm_v_{i}").value = f["value"]
            getattr(model, f"fwhm_v_{i}").bounds = f["bounds"]
            getattr(model, f"fwhm_v_{i}").fixed = f["fixed"]
            getattr(model, f"fwhm_v_{i}").tied = f["tied"]

            getattr(model, f"v_off_{i}").value = v["value"]
            getattr(model, f"v_off_{i}").bounds = v["bounds"]
            getattr(model, f"v_off_{i}").fixed = v["fixed"]
            getattr(model, f"v_off_{i}").tied = v["tied"]

        return model
