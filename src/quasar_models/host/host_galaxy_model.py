"""
AstroPy compatible model: HostGalaxyModel.
"""
from logging import getLogger
from typing import Any, ClassVar, Literal, Self

import numpy as np
from astropy.constants import c
from astropy.modeling import Parameter
from pydantic import validate_call
from quasar_typing.bounds import AstropyBounds
from quasar_typing.literals import HostGalaxyTemplateName
from quasar_typing.numpy import FloatVector
from quasar_utils.setup import Info

from quasar_models._core.modeling.host import (
    HostGalaxyEvaluate,
    HostGalaxyFitDeriv,
)
from quasar_models.modeling.template import TemplateModel

from .host_galaxy_template import HostGalaxyTemplate
from .io import convert_params_to_name

logger = getLogger(__name__)

C_KMS: float = c.to("km/s").value  # Exact speed of light in km/s


class HostGalaxyModel(TemplateModel):
    flux = Parameter(
        description="Flux density at ('fwhm_norm', 'x_norm')",
        default=1.0,
        min=0.0,
    )
    fwhm = Parameter(
        description="FWHM (km/s) of the template",
        default=0.0,
        min=0.0,
        fixed=True,
    )

    model_type: ClassVar[Literal["hg"]] = "hg"

    TEMPLATE_KEYS: ClassVar[tuple[Literal["template"]]] = ("template",)
    CYTEMPLATE_KEYS: ClassVar[tuple[Literal["cytemplate"]]] = ("cytemplate",)

    @classmethod
    @validate_call
    def create(
        cls,
        flux: float,
        fwhm: float,
        *,
        info: Info | None = None,
        template: HostGalaxyTemplate | None = None,
        allow_interp_fitting: bool = False,
        n_scales: float = 3.0,
        name: HostGalaxyTemplateName | None = None,
        age: int | None = None,
        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        flux_fixed: bool | None = None,
        fwhm_fixed: bool | None = None,
    ) -> Self:
        from quasar_models.utils.astropy import _apply_bounds

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
        if flux_bounds is not None:
            model.flux.value = _apply_bounds(flux, flux_bounds)
            model.flux.bounds = flux_bounds

        if fwhm_bounds is None and template.fwhm.shape[0] > 1:
            fwhm_bounds = (template.fwhm[0], template.fwhm[-1])

        if fwhm_bounds is not None:
            model.fwhm.value = _apply_bounds(fwhm, fwhm_bounds)
            model.fwhm.bounds = fwhm_bounds

        if flux_fixed is not None:
            model.flux.fixed = flux_fixed
        if fwhm_fixed is not None:
            model.fwhm.fixed = fwhm_fixed

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
        return (3.0, 0.0)

    ###

    def evaluate(self, x, flux, fwhm, y=None):
        return self.evaluate_func(
            x,
            *self._transform_args_if_ndarray(flux, fwhm),
            **self.kwargs,
            y=y,
        )

    def partial_deriv(self, x, flux, fwhm, derivs=None):
        return self.fit_deriv_func(
            x,
            *self._transform_args_if_ndarray(flux, fwhm),
            **self.kwargs,
            derivs=derivs,
        )

    def fit_deriv(self, x, flux, fwhm, derivs=None):
        return list(self.partial_deriv(x, flux, fwhm, derivs=derivs))

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.meta.get(
            "kwargs",
            {
                "template": self.template,
                "n_scales": self.n_scales,
                "interpolation_matrix": self.interpolation_matrix,
            },
        )

    @kwargs.setter
    def kwargs(self, value: dict[str, Any]) -> None:
        self.meta["kwargs"] = value

    @kwargs.deleter
    def kwargs(self) -> None:
        self.meta.pop("kwargs", None)

    def _set_kwargs(self) -> None:
        self.kwargs = {
            "template": self.cytemplates["cytemplate"],
            "n_scales": self.n_scales,
            "interpolation_matrix": self.interpolation_matrix,
        }

    ### Model preparation

    @property
    def evaluate_func(self) -> HostGalaxyEvaluate:
        from quasar_models._core.modeling.host import evaluate_exact

        return self.meta.get("evaluate_func", evaluate_exact)

    @evaluate_func.setter
    def evaluate_func(self, value: HostGalaxyEvaluate) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> HostGalaxyFitDeriv:
        from quasar_models._core.modeling.host import fit_deriv_exact_all

        return self.meta.get("fit_deriv_func", fit_deriv_exact_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: HostGalaxyFitDeriv) -> None:
        self.meta["fit_deriv_func"] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop("fit_deriv_func", None)

    def _choose_evaluate_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        from quasar_models._core.modeling.host import choose_evaluate_func

        self.evaluate_func = choose_evaluate_func(
            self.allow_interp_fitting,
            fixed or self.fixed,
        )

    def _choose_fit_deriv_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        from quasar_models._core.modeling.host import choose_fit_deriv_func

        self.fit_deriv_func = choose_fit_deriv_func(
            self.allow_interp_fitting,
            fixed or self.fixed,
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
        from quasar_utils.raster import rasterise

        template = self.template.interpolate(x)
        chi2s, fluxs = rasterise(
            y,
            dy,
            template.fwhm,
            template.data / template.normalisation,
            flux_bounds=self.flux.bounds,
            fwhm_bounds=self.fwhm.bounds,
        )
        obj = self if inplace else self.copy()

        if not np.isfinite(chi2s).any():
            return obj

        idx: int = np.argmin(chi2s).flatten()[0]
        chi2 = chi2s[idx]
        flux = fluxs[idx]
        fwhm = self.template.fwhm[idx]

        msg = f"Raster fit results: {chi2=:.1f}, {flux=:.1e}, {fwhm=:.1e}."
        logger.debug(msg)

        obj.flux.value = flux
        obj.fwhm.value = fwhm

        return obj


type HgModel = HostGalaxyModel