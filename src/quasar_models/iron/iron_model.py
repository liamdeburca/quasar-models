"""
AstroPy compatible model: IronModel.
"""

from logging import getLogger
from typing import Any, ClassVar, Literal, Self

from astropy.modeling import Parameter
from numpy import (
    argmin,
    array_equal,
    float64,
    inf,
    invert,
    isfinite,
    zeros_like,
)
from quasar_typing.bounds import AstropyBounds
from quasar_typing.numpy import FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.raster import rasterise
from quasar_utils.setup import Info

from quasar_models._core.modeling.iron import (
    IronEvaluate,
    IronFitDeriv,
    choose_evaluate_func,
    choose_fit_deriv_func,
    evaluate_exact,
    fit_deriv_exact_all,
)
from quasar_models.modeling.template import TemplateModel

from ..utils.astropy import apply_bounds
from ..utils.serialization import deserialize_parameter, serialize_parameter
from .iron_template import IronTemplate

logger = getLogger(__name__)


class IronModel(TemplateModel):
    flux = Parameter(
        default=1.0,
        min=0.0,
        fixed=False,
    )
    fwhm = Parameter(
        default=1.0,
        min=0.0,
        fixed=False,
    )
    split = Parameter(
        default=1.0,
        min=0.0,
        fixed=True,
    )
    left = Parameter(
        default=1.0,
        min=0.0,
        max=1.0,
        fixed=True,
    )
    right = Parameter(
        default=1.0,
        min=0.0,
        max=1.0,
        fixed=True,
    )

    model_type: ClassVar[Literal["fe"]] = "fe"

    TEMPLATE_KEYS: ClassVar[tuple[Literal["template"]]] = ("template",)
    CYTEMPLATE_KEYS: ClassVar[tuple[Literal["cytemplate"]]] = ("cytemplate",)

    @classmethod
    def create(
        cls,
        flux: float,
        fwhm: float,
        *,
        scale: float,
        template: IronTemplate | None = None,
        info: Info | None = None,
        split: float = 1.0,
        left: float = 1.0,
        right: float = 1.0,
        allow_interp_fitting: bool = False,
        n_scales: float | None = None,
        name: str | Literal["vw2001", "v2003", "bw"] | None = None,

        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        split_bounds: AstropyBounds | None = None,
        left_bounds: AstropyBounds | None = None,
        right_bounds: AstropyBounds | None = None,

        flux_fixed: bool | None = None,
        fwhm_fixed: bool | None = None,
        split_fixed: bool | None = None,
        left_fixed: bool | None = None,
        right_fixed: bool | None = None,
    ) -> Self:
        if template is None:
            if info is None:
                msg = "'info' cannot be None if 'template' is not provided."
                raise ValueError(msg)
            if name is None:
                msg = "'name' cannot be None if 'template' is not provided."
                raise ValueError(msg)

            template = IronTemplate.load_from_cache(name=name, info=info)

        if n_scales is not None:
            template.n_scales = n_scales

        model = IronModel(
            flux,
            fwhm,
            split=split,
            left=left,
            right=right,
            name=template.name,
            meta={
                "template": template,
                "scale": scale,
                "allow_interp_fitting": allow_interp_fitting,
                "_interpolation_matrices": {},
            },
        )
        if flux_bounds is not None:
            model.flux.value = apply_bounds.__wrapped__(flux, flux_bounds)
            model.flux.bounds = flux_bounds

        if fwhm_bounds is None and template.fwhm.shape[0] > 1:
            fwhm_bounds = (template.fwhm[0], template.fwhm[-1])
        if fwhm_bounds is not None:
            model.fwhm.value = apply_bounds.__wrapped__(fwhm, fwhm_bounds)
            model.fwhm.bounds = fwhm_bounds

        if split_bounds is None:
            split_bounds = (template.x[0], template.x[-1])
        if split_bounds is not None:
            model.split.value = apply_bounds.__wrapped__(split, split_bounds)
            model.split.bounds = split_bounds

        if left_bounds is not None:
            model.left.value = apply_bounds.__wrapped__(left, left_bounds)
            model.left.bounds = left_bounds
        if right_bounds is not None:
            model.right.value = apply_bounds.__wrapped__(right, right_bounds)
            model.right.bounds = right_bounds

        if flux_fixed is not None:
            model.flux.fixed = flux_fixed
        if fwhm_fixed is not None:
            model.fwhm.fixed = fwhm_fixed
        if split_fixed is not None:
            model.split.fixed = split_fixed
        if left_fixed is not None:
            model.left.fixed = left_fixed
        if right_fixed is not None:
            model.right.fixed = right_fixed

        return model

    @property
    def template(self) -> IronTemplate:
        return self.meta["template"]

    @template.setter
    def template(self, value: IronTemplate) -> None:
        self.meta["template"] = value

    @property
    def n_scales(self) -> float:
        return self.template.n_scales

    @n_scales.setter
    def n_scales(self, value: float) -> None:
        self.template.n_scales = value

    @property
    def scale(self) -> float:
        return self.meta["scale"]

    @scale.setter
    def scale(self, value: float) -> None:
        self.meta["scale"] = value

    def evaluate(self, x, flux, fwhm, split, left, right, y=None):
        return self.evaluate_func(
            x,
            *self._transform_args_if_ndarray(flux, fwhm, split, left, right),
            **self.kwargs,
            y=y,
        )

    def partial_deriv(self, x, flux, fwhm, split, left, right, derivs=None):
        return self.fit_deriv_func(
            x,
            *self._transform_args_if_ndarray(flux, fwhm, split, left, right),
            **self.kwargs,
            derivs=derivs,
        )

    def fit_deriv(self, x, flux, fwhm, split, left, right, derivs=None):
        return list(self.partial_deriv(x, flux, fwhm, split, left, right, derivs=derivs))

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.meta.get(
            "kwargs",
            {
                "template": self.template,
                "scale": self.scale,
                "n_scales": self.n_scales,
                "interpolation_matrix": self.interpolation_matrix,
            }
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
            "scale": self.scale,
            "n_scales": self.n_scales,
            "interpolation_matrix": self.interpolation_matrix,
        }

    ### Model preparation

    @property
    def evaluate_func(self) -> IronEvaluate:
        return self.meta.get("evaluate_func", evaluate_exact)

    @evaluate_func.setter
    def evaluate_func(self, value: IronEvaluate) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> IronFitDeriv:
        return self.meta.get("fit_deriv_func", fit_deriv_exact_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: IronFitDeriv) -> None:
        self.meta["fit_deriv_func"] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop("fit_deriv_func", None)

    def _choose_evaluate_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self.evaluate_func = choose_evaluate_func(
            self.left.value,
            self.right.value,
            self.allow_interp_fitting,
            fixed or self.fixed,
        )

    def _choose_fit_deriv_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.left.value,
            self.right.value,
            self.allow_interp_fitting,
            fixed or self.fixed,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (1.0, self.template.x_norm)

    # Utilities

    @validate_call
    def rasterFit(
        self,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        *,
        bias: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:
        """
        For each available FWHM, calculates the best-fit flux and the
        corresponding goodness-of-fit (chi-square), identifying the best
        flux-FWHM pair.
        """
        if not inplace:
            cp = self.copy()
            cp.rasterFit.__wrapped__(cp, x, y, dy, bias=bias, inplace=True)
            return cp

        template = self.template \
            if array_equal(x, self.template.x) \
            else self.template.interpolate(x)

        data = template.data / template.normalisation

        if self.left.fixed and self.right.fixed:
            # Perform a single raster fit
            if self.left.value != 1.0 or self.right.value != 1.0:
                data = data[:1,:]
                data *= template._get_split_weight(
                    x,
                    self.split.value,
                    self.left.value,
                    self.right.value,
                    self.scale,
                )[None,:]

            chi2s, fluxs = rasterise.__wrapped__(
                y,
                dy,
                template.fwhm,
                data,
                flux_bounds=self.flux.bounds,
                fwhm_bounds=self.fwhm.bounds,
            )

            if not isfinite(chi2s).any():
                # Failed rasterisation, possibly due to selected data not
                # covering template.
                return self

            idx: int = argmin(chi2s)
            self.flux.value = fluxs[idx]
            self.fwhm.value = template.fwhm[idx]
        else:
            # Perform separate raster fits for left and right halves
            is_left = self.template.x < self.split.value
            is_right = invert(is_left)

            mask_left = x < self.split.value
            mask_right = invert(mask_left)

            chi2s = zeros_like(self.template.fwhm, dtype=float64)
            if cond_l := (is_left.any() and (mask_left.sum() >= 2)):
                _y = y[mask_left]
                _dy = dy[mask_left]
                _data = data[:,mask_left]

                chi2s_left, fluxs_left = rasterise.__wrapped__(
                    _y, _dy, self.template.fwhm, _data,
                    flux_bounds=self.flux.bounds,
                    fwhm_bounds=self.fwhm.bounds,
                )
                chi2s += chi2s_left
            if cond_r := (is_right.any() and (mask_right.sum() >= 2)):
                _y = y[mask_right]
                _dy = dy[mask_right]
                _data = data[:,mask_right]
                chi2s_right, fluxs_right = rasterise.__wrapped__(
                    _y, _dy, self.template.fwhm, _data,
                    flux_bounds=self.flux.bounds,
                    fwhm_bounds=self.fwhm.bounds,
                )
                chi2s += chi2s_right

            chi2s[chi2s == 0] = inf
            idx: int = argmin(chi2s)

            self.flux.value = fluxs[idx]
            self.fwhm.value = apply_bounds(template.fwhm[idx], self.fwhm.bounds)

            if bias == "left":
                self.left.value = 1.0
                self.left.bounds = (min(self.left.bounds[0], 1.0), 1.0)
                self.left.fixed = True

                flux = (fluxs_left if cond_l else fluxs_right)[idx]
                right = fluxs_right[idx] / flux if cond_r else 1.0

                self.right.value = apply_bounds(right, self.right.bounds)
                self.right.fixed = False

            else:
                self.right.value = 1.0
                self.right.bounds = (min(self.right.bounds[0], 1.0), 1.0)
                self.right.fixed = True

                flux = (fluxs_right if cond_r else fluxs_left)[idx]
                left = fluxs_left[idx] / flux if cond_l else 1.0

                self.left.value = apply_bounds(left, self.left.bounds)
                self.left.fixed = False

        return self

    ### Serialization

    def serialize(self, info: Info) -> dict[str, dict[str, Any]]:
        kms_unit = "km/s"
        flux_unit = str(info.units.flux_unit)
        wave_unit = str(info.units.wavelength_unit)
        data = {
            "name": self.name,
            "flux": serialize_parameter(self.flux, flux_unit),
            "fwhm": serialize_parameter(self.fwhm, kms_unit),
            "split": serialize_parameter(self.split, wave_unit),
            "left": serialize_parameter(self.left, None),
            "right": serialize_parameter(self.right, None),
            "template": self.template.serialize(info),
        }
        return {f"IronModel::{self.name}": data}

    @classmethod
    def deserialize(cls, data: dict[str, Any], name: str, info: Info) -> Self:
        kms_unit = "km/s"
        flux_unit = str(info.units.flux_unit)
        wave_unit = str(info.units.wavelength_unit)

        flux = deserialize_parameter(data["flux"], flux_unit)
        fwhm = deserialize_parameter(data["fwhm"], kms_unit)
        split = deserialize_parameter(data["split"], wave_unit)
        left = deserialize_parameter(data["left"], None)
        right = deserialize_parameter(data["right"], None)
        template = IronTemplate.deserialize(data["template"], info)

        model = IronModel.create(
            flux["value"], fwhm["value"],
            scale=info.iron.scale,
            template=template,
            info=info,
            split=split["value"],
            left=left["value"],
            right=right["value"],
            allow_interp_fitting=info.convolution.allow_interp_fitting,
            flux_bounds=flux["bounds"],
            fwhm_bounds=fwhm["bounds"],
            split_bounds=split["bounds"],
            left_bounds=left["bounds"],
            right_bounds=right["bounds"],
            flux_fixed=flux["fixed"],
            fwhm_fixed=fwhm["fixed"],
            split_fixed=split["fixed"],
            left_fixed=left["fixed"],
            right_fixed=right["fixed"],
        )
        model.flux.tied = flux["tied"]
        model.fwhm.tied = fwhm["tied"]
        model.split.tied = split["tied"]
        model.left.tied = left["tied"]
        model.right.tied = right["tied"]

        model.name = data["name"]

        return model
