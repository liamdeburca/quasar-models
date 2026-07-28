"""
AstroPy compatible model: HostGalaxyModel.
"""

from logging import getLogger
from typing import Any, ClassVar, Literal, Self

from astropy.modeling import Parameter
from numpy import argmin, isfinite
from quasar_typing.numpy import FloatMatrix, FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from quasar_models._core.modeling.host import (
    HostGalaxyEvaluate,
    HostGalaxyFitDeriv,
    choose_evaluate_func,
    choose_fit_deriv_func,
    evaluate_exact,
    fit_deriv_exact_all,
)
from quasar_models.modeling.template import TemplateModel

from ..utils.astropy import apply_bounds
from .host_galaxy_template import HostGalaxyTemplate
from .io import convert_params_to_name

logger = getLogger(__name__)


class HostGalaxyModel(TemplateModel):
    flux = Parameter(
        default=1.0,
        min=0.0,
    )
    fwhm = Parameter(
        default=0.0,
        min=0.0,
        fixed=True,
    )

    model_type: ClassVar[Literal["hg"]] = "hg"

    TEMPLATE_KEYS: ClassVar[tuple[Literal["template"]]] = ("template",)
    CYTEMPLATE_KEYS: ClassVar[tuple[Literal["cytemplate"]]] = ("cytemplate",)

    @classmethod
    def create(
        cls,
        flux: float,
        fwhm: float,
        *,
        info: Info | None = None,
        template: HostGalaxyTemplate | None = None,
        allow_interp_fitting: bool = False,
        n_scales: float = 3.0,
        name: Literal["bc2003"] | None = None,
        age: int | None = None,
    ) -> Self:
        if template is None:
            if info is None:
                msg = "'info' cannot be None if 'template' is not provided."
                raise ValueError(msg)
            if name is None:
                msg = "'name' cannot be None if 'template' is not provided."
                raise ValueError(msg)
            if age is None:
                msg = "'age' cannot be None if 'template' is not provided."
                raise ValueError(msg)

            template = HostGalaxyTemplate.load_from_cache(
                name=name,
                age=age,
                info=info,
            )

        model = HostGalaxyModel(
            flux,
            fwhm,
            name=convert_params_to_name(template.name, template.age),
            meta={
                "template": template,
                "allow_interp_fitting": allow_interp_fitting,
                "_interpolation_matrices": {},
            },
        )
        model.fwhm.bounds = (template.fwhm[0], template.fwhm[-1])
        model.fwhm.value = apply_bounds.__wrapped__(
            model.fwhm.value,
            model.fwhm.bounds,
        )

        return model

    @property
    def template(self) -> HostGalaxyTemplate:
        return self.meta["template"]

    @template.setter
    def template(self, value: HostGalaxyTemplate) -> None:
        self.meta["template"] = value

    @property
    def n_scales(self) -> float:
        return self.template.n_scales

    @n_scales.setter
    def n_scales(self, value: float) -> None:
        self.template.n_scales = value

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (3.0, self.template.x_norm)

    ###

    def evaluate(
        self,
        x: float | FloatVector,
        flux: float,
        fwhm: float,
        y: FloatVector | None = None,
    ) -> float | FloatVector:
        return self.evaluate_func(
            x,
            flux,
            fwhm,
            **self._kwargs,
            y=y,
        )

    def partial_deriv(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        return self.fit_deriv_func(
            x,
            flux,
            fwhm,
            **self._kwargs,
            derivs=derivs,
        )

    def fit_deriv(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        derivs: FloatMatrix | None = None,
    ) -> list[FloatVector]:
        return list(self.partial_deriv(x, flux, fwhm, derivs=derivs))

    @property
    def _kwargs(self) -> dict[str, Any]:
        cytemplates = self.cytemplates
        return {
            "template": self.template,
            "cytemplate": cytemplates["cytemplate"],
            "n_scales": self.n_scales,
            "interpolation_matrix": self.interpolation_matrix,
        }

    ### Model preparation

    @property
    def evaluate_func(self) -> HostGalaxyEvaluate:
        return self.meta.get("evaluate_func", evaluate_exact)

    @evaluate_func.setter
    def evaluate_func(self, value: HostGalaxyEvaluate) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> HostGalaxyFitDeriv:
        return self.meta.get("fit_deriv_func", fit_deriv_exact_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: HostGalaxyFitDeriv) -> None:
        self.meta["fit_deriv_func"] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop("fit_deriv_func", None)

    def _choose_evaluate_func(self) -> None:
        self.evaluate_func = choose_evaluate_func(
            self.allow_interp_fitting,
        )

    def _choose_fit_deriv_func(self) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.allow_interp_fitting,
            self.fixed,
        )

    # Utilities

    @validate_call
    def rasterFit(
        self,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        *,
        inplace: bool = False,
    ) -> Self:
        chi2s, fluxs = self.template.rasterise.__wrapped__(
            self.template,
            x,
            y,
            dy,
            flux_bounds=self.flux.bounds,
            fwhm_bounds=self.fwhm.bounds,
        )
        obj = self if inplace else self.copy()

        if not isfinite(chi2s).any():
            return obj

        idx: int = argmin(chi2s).flatten()[0]
        chi2 = chi2s[idx]
        flux = fluxs[idx]
        fwhm = self.template.fwhm[idx]

        msg = f"Raster fit results: {chi2=:.1f}, {flux=:.1e}, {fwhm=:.1e}."
        logger.debug(msg)

        obj.flux.value = flux
        obj.fwhm.value = fwhm

        return obj
