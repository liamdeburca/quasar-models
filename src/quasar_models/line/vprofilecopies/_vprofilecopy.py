__all__ = [
    "_VProfileCopy",
]

from collections.abc import Callable, Iterable
from itertools import product
from typing import ClassVar, Literal, Self, Union

from numpy import array, float64
from quasar_typing.astropy import CompoundModel_

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
from quasar_models.utils.astropy import apply_bounds


def get_params_from_args(n_profiles: int, *args) -> tuple:
    return tuple(
        array(
            [float(args[3 * i + o]) for i in range(n_profiles)],
            dtype=float64,
            order="C",
        )
        for o in range(3)
    )


###


class _VProfileCopy(BaseModel):
    n_profiles: ClassVar[Literal[1, 2, 3, 4, 5]]
    model_type: ClassVar[Literal["em"]] = "em"

    @property
    def pure_name(self) -> str:
        return self.name

    @property
    def wave(self) -> float:
        return self.meta["wave"]

    @wave.setter
    def wave(self, value: float) -> None:
        self.meta["wave"] = value

    @property
    def sigma_res(self) -> float | None:
        return self.meta["sigma_res"]

    @sigma_res.setter
    def sigma_res(self, value: float | None) -> None:
        self.meta["sigma_res"] = value

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
        *gs: GaussianModel,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,
        sigma_res: float | None = None,
        master_name: str | None = None,
        n_sigmas: float | None = None,
        adapt: bool = True,
        freeze: bool = False,
    ) -> Self:
        meta = cls._get_metadata(
            wave,
            name,
            *gs,
            sigma_res=sigma_res,
            master_name=master_name,
            n_sigmas=n_sigmas,
        )
        model = cls(
            strength_scale_value,
            name=name,
            meta=meta,
        )
        model.strength_scale.value = strength_scale_value
        model.strength_scale.bounds = strength_scale_bounds
        model.strength_scale.fixed = strength_scale_fixed

        if adapt:
            model._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        if freeze:
            model._freeze_velocity_profile(inplace=True)
            model._forget_ties(inplace=True)

        return model

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: Union[GaussianModel, Iterable[GaussianModel]],
        *,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,
        adapt: bool = True,
        freeze: bool = False,
    ) -> Self:
        if cls.n_profiles == 1:
            assert isinstance(model, GaussianModel)
            model = (model,)
        if cls.n_profiles > 1:
            model = tuple(model)
            assert len(model) == cls.n_profiles

        return cls.create(
            wave,
            name,
            *model,
            strength_scale_value=strength_scale_value,
            strength_scale_bounds=strength_scale_bounds,
            strength_scale_fixed=strength_scale_fixed,
            adapt=adapt,
            freeze=freeze,
        )

    def makeCopy(
        self,
        master_model: Union[GaussianModel, Iterable[GaussianModel]],
        *,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,
        freeze: bool = False,
    ) -> Self:
        return self.from_model(
            self.wave,
            self.name,
            master_model,
            strength_scale_value=strength_scale_value,
            strength_scale_bounds=strength_scale_bounds,
            strength_scale_fixed=strength_scale_fixed,
            adapt=True,
            freeze=freeze,
        )

    @classmethod
    def _get_metadata(
        cls,
        wave: float,
        name: str,
        *gs: GaussianModel,
        sigma_res: float | None = None,
        master_name: str | None = None,
        n_sigmas: float | None = None,
    ) -> dict:
        meta = {
            "pure_name": name,
            "wave": wave,
        }

        if len(gs) == 0:
            # Used when copying
            meta["sigma_res"] = sigma_res
            meta["master_name"] = master_name
            meta["n_sigmas"] = n_sigmas
        elif len(gs) == cls.n_profiles:
            meta["sigma_res"] = gs[0].sigma_res
            meta["master_name"] = gs[0].pure_name
            meta["n_sigmas"] = gs[0].n_sigmas

        return meta

    def __str__(self) -> str:
        name = self.name
        master = self.master_name
        wave = self.wave
        untied = hasattr(self, "_prev_ties")
        return (
            f"{self.__class__.__name__}({name=}, {master=}, {wave=}, untied={untied})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def evaluate(self, x, strength_scale, *args):
        return self.evaluate_func(
            x,
            strength_scale,
            *get_params_from_args(self.n_profiles, *args),
            **self._kwargs,
            y=None,
        )

    def jac(self, x, strength_scale, *args):
        return self.fit_deriv_func(
            x,
            strength_scale,
            *get_params_from_args(self.n_profiles, *args),
            **self._kwargs,
            derivs=None,
        )

    def fit_deriv(self, x, strength_scale, *args):
        return list(self.jac(x, strength_scale, *args))

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
        self.evaluate_func = choose_evaluate_func()

    def _choose_fit_deriv_func(self) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.fixed_dict or self.fixed,
        )

    @property
    def _kwargs(self) -> dict:
        return {
            "wave": self.wave,
            "sigma_res": self.sigma_res,
        }

    ###

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (4.0, self.wave)

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

    def _freeze_velocity_profile(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Fixes all profile parameters: 'strength', 'fwhm_v', and 'v_off'.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies this instance. If False, returns a modified copy.
            Defaults to False.

        Returns
        -------
        _VProfileCopy
            Modified instance (self if inplace=True, otherwise a copy).

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        for i, pname in product(
            range(1, model.n_profiles + 1),
            ("strength", "fwhm_v", "v_off"),
        ):
            getattr(model, f"{pname}_{i}").fixed = True

        return model

    def _thaw_velocity_profile(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Unfixes all profile parameters: 'strength', 'fwhm_v', and 'v_off'.

        Parameters
        ----------
        inplace : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        for i, pname in product(
            range(1, model.n_profiles + 1),
            ("strength", "fwhm_v", "v_off"),
        ):
            getattr(model, f"{pname}_{i}").fixed = False

        return model

    def _forget_ties(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Removes all profile ties ('strength', 'fwhm_v', and 'v_off') and
        stores them in the '_prev_ties' attribute.

        Parameters
        ----------
        inplace : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        if not hasattr(model, "_prev_ties"):
            _prev_ties: dict[str, Callable] = {}

            for i, pname in product(
                range(1, model.n_profiles + 1),
                ("strength", "fwhm_v", "v_off"),
            ):
                attr_name = f"{pname}_{i}"
                attr = getattr(model, attr_name)
                if tie := attr.tied:
                    _prev_ties[attr_name] = tie
                    attr.tied = False

            model._prev_ties = _prev_ties

        return model

    def _remember_ties(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Restores all profile ties ('strength', 'fwhm_v', and 'v_off') from the
        '_prev_ties' attribute and deletes '_prev_ties'.

        Parameters
        ----------
        inplace : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        if hasattr(model, "_prev_ties"):
            for attr_name, tie in model._prev_ties.items():
                getattr(model, attr_name).tied = tie

            del model._prev_ties

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
            g = GaussianModel(self.wave, self.sigma_res, name=name)

            for attr_name in ("strength", "fwhm_v", "v_off"):
                getattr(g, attr_name).value = getattr(
                    self, f"{attr_name}_{i + 1}"
                ).value
                getattr(g, attr_name).bounds = getattr(
                    self, f"{attr_name}_{i + 1}"
                ).bounds

            gaussian_models.append(g)

        return gaussian_models
