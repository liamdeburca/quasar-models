"""
AstroPy compatible model: BalmerModel.
"""

from typing import Any, ClassVar, Literal, Self

from astropy.constants import c
from astropy.modeling import Parameter
from astropy.units import Unit
from numpy import (
    argmin,
    array,
    array_equal,
    concatenate,
    float64,
    inf,
    unique,
)
from numpy.typing import NDArray
from quasar_typing.bounds import AstropyBounds
from quasar_typing.numpy import FittableFloatVector, FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.interpolation import create_interp_matrix
from quasar_utils.raster import rasterise
from quasar_utils.setup import Info

from quasar_models._core.modeling.balmer import (
    BalmerEvaluate,
    BalmerFitDeriv,
    choose_evaluate_func,
    choose_fit_deriv_func,
    evaluate_exact,
    fit_deriv_exact_all,
)
from quasar_models.modeling.template import TemplateModel

from ..continuum import PowerLawModel
from ..utils.astropy import apply_bounds
from ..utils.serialization import (
    deserialize_parameter,
    deserialize_quantity,
    serialize_parameter,
    serialize_quantity,
)
from .continuum import BalmerContinuumTemplate
from .series import BalmerSeriesTemplate

C_KMS: float = c.to("km/s").value


class BalmerModel(TemplateModel):
    flux = Parameter(
        description="Flux density at ('fwhm_norm', 'x_norm')",
        default=1.0,
        min=0.0,
    )
    fwhm = Parameter(
        description="FWHM (km/s) of the template",
        default=0.0,
        min=0.0,
    )
    ratio = Parameter(
        description="Continuum-to-series flux ratio",
        default=1.0,
        min=0.0,
    )

    model_type: ClassVar[Literal["ba"]] = "ba"

    TEMPLATE_KEYS: ClassVar[tuple[str, str]] = (
        "continuum_template",
        "series_template",
    )
    CYTEMPLATE_KEYS: ClassVar[tuple[str, str]] = (
        "continuum_cytemplate",
        "series_cytemplate",
    )

    @classmethod
    def create(
        cls,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        edge: float,
        continuum_template: BalmerContinuumTemplate | None = None,
        series_template: BalmerSeriesTemplate | None = None,
        info: Info | None = None,
        temp: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        dens: float | None = None,
        n_u_range: tuple[int, int] | None = None,
        allow_interp_fitting: bool = False,
        n_scales: float | None = None,
        name: str | Literal["SH1995"] | None = None,

        flux_bounds: AstropyBounds | None = None,
        fwhm_bounds: AstropyBounds | None = None,
        ratio_bounds: AstropyBounds | None = None,
        flux_fixed: bool | None = None,
        fwhm_fixed: bool | None = None,
        ratio_fixed: bool | None = None,

    ) -> Self:
        if continuum_template is None:
            if info is None:
                msg = "'info' cannot be None if 'continuum_template' is not provided."
                raise ValueError(msg)
            if temp is None:
                msg = "'temp' cannot be None if 'continuum_template' is not provided."
                raise ValueError(msg)
            if tau is None:
                msg = "'tau' cannot be None if 'continuum_template' is not provided."
                raise ValueError(msg)
            if scale is None:
                msg = "'scale' cannot be None if 'continuum_template' is not provided."
                raise ValueError(msg)

            continuum_template = BalmerContinuumTemplate.load_from_cache(
                temp=temp,
                tau=tau,
                scale=scale,
                info=info,
            )

        if series_template is None:
            if info is None:
                msg = "'info' cannot be None if 'series_template' is not provided."
                raise ValueError(msg)
            if name is None:
                msg = "'name' cannot be None if 'series_template' is not provided."
                raise ValueError(msg)
            if temp is None:
                msg = "'temp' cannot be None if 'series_template' is not provided."
                raise ValueError(msg)
            if dens is None:
                msg = "'dens' cannot be None if 'series_template' is not provided."
                raise ValueError(msg)
            if n_u_range is None:
                msg = "'n_u_range' cannot be None if 'series_template' is not provided."
                raise ValueError(msg)

            series_template = BalmerSeriesTemplate.load_from_cache(
                name=name,
                temp=temp,
                dens=dens,
                n_u_range=n_u_range,
                info=info,
            )

        if not array_equal(continuum_template.x, series_template.x):
            msg = "The continuum and series templates do not have the same x arrays!"
            raise ValueError(msg)

        if continuum_template.temp != series_template.temp:
            msg = (
                f"The continuum ({continuum_template.temp}) and series ({series_template.temp}) template do not have "
                "the same temperatures!"
            )
            raise ValueError(msg)

        if n_scales is not None:
            continuum_template.n_scales = n_scales
            series_template.n_scales = n_scales
        elif continuum_template.n_scales != series_template.n_scales:
            msg = (
                f"The continuum ({continuum_template.n_scales}) and series ({series_template.n_scales}) template do not have "
                "the same 'n_scales' values! Setting according to maximum "
                "value."
            )
            # logger.info(msg)
            n_scales = max(continuum_template.n_scales, series_template.n_scales)
            continuum_template.n_scales = n_scales
            series_template.n_scales = n_scales

        if not array_equal(continuum_template.fwhm, series_template.fwhm):
            fwhms = unique(
                concatenate([continuum_template.fwhm, series_template.fwhm]),
            )
            continuum_template.upsample(fwhms, inplace=True)
            series_template.upsample(fwhms, inplace=True)

        model = BalmerModel(
            flux, fwhm, ratio,
            name=name or "balmer",
            meta={
                "continuum_template": continuum_template,
                "series_template": series_template,
                "allow_interp_fitting": allow_interp_fitting,
                "edge": edge,
            },
        )
        if flux_bounds is not None:
            model.flux.value = apply_bounds.__wrapped__(flux, flux_bounds)
            model.flux.bounds = flux_bounds

        if fwhm_bounds is None:
            model.fwhm.bounds = (
                continuum_template.fwhm[0], 
                continuum_template.fwhm[-1],
            )
        model.fwhm.value = apply_bounds.__wrapped__(fwhm, fwhm_bounds)
        model.fwhm.bounds = fwhm_bounds

        if ratio_bounds is not None:
            model.ratio.value = apply_bounds.__wrapped__(ratio, ratio_bounds)
            model.ratio.bounds = ratio_bounds

        if flux_fixed is not None:
            model.flux.fixed = flux_fixed
        if fwhm_fixed is not None:
            model.fwhm.fixed = fwhm_fixed
        if ratio_fixed is not None:
            model.ratio.fixed = ratio_fixed

        return model

    # Continuum Template

    @property
    def continuum_template(self) -> BalmerContinuumTemplate | None:
        return self.meta.get("continuum_template", None)

    @continuum_template.setter
    def continuum_template(self, value: BalmerContinuumTemplate) -> None:
        self.meta["continuum_template"] = value

    @continuum_template.deleter
    def continuum_template(self) -> None:
        self.meta.pop("continuum_template", None)

    # Series Template

    @property
    def series_template(self) -> BalmerSeriesTemplate | None:
        return self.meta.get("series_template", None)

    @series_template.setter
    def series_template(self, value: BalmerSeriesTemplate) -> None:
        self.meta["series_template"] = value

    @series_template.deleter
    def series_template(self) -> None:
        self.meta.pop("series_template", None)

    # Other Properties

    @property
    def n_scales(self) -> float:
        return self.continuum_template.n_scales

    @n_scales.setter
    def n_scales(self, value: float) -> None:
        self.continuum_template.n_scales = value
        self.series_template.n_scales = value

    @property
    def edge(self) -> float:
        return self.meta["edge"]

    @edge.setter
    def edge(self, value: float) -> None:
        self.meta["edge"] = value

    @property
    def waves(self) -> FloatVector:
        return self.series_template.waves

    @property
    def weights(self) -> FloatVector:
        return self.series_template.weights

    @property
    def temp(self) -> float:
        return self.series_template.temp

    @property
    def dens(self) -> float:
        return self.series_template.dens

    @property
    def n_u_range(self) -> tuple[int, int]:
        return self.series_template.n_u_range

    @property
    def source(self) -> str:
        return self.series_template.name

    @property
    def tau(self) -> float:
        return self.continuum_template.tau

    @property
    def scale(self) -> float:
        return self.continuum_template.scale

    def evaluate(self, x,flux, fwhm, ratio, y=None):
        return self.evaluate_func(
            x,
            *self._transform_args_if_ndarray(flux, fwhm, ratio),
            **self.kwargs,
            y=y,
        )

    def partial_deriv(self, x, flux, fwhm, ratio, derivs=None):
        return self.fit_deriv_func(
            x,
            *self._transform_args_if_ndarray(flux, fwhm, ratio),
            **self.kwargs,
            derivs=derivs,
        )

    def fit_deriv(self, x, flux, fwhm, ratio, derivs=None):
        return list(self.partial_deriv(x, flux, fwhm, ratio, derivs=derivs))

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.meta.get(
            "kwargs",
            {
                "continuum_template": self.continuum_template,
                "series_template": self.series_template,
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
            "continuum_template": self.cytemplates["continuum_cytemplate"],
            "series_template": self.cytemplates["series_cytemplate"],
            "n_scales": self.n_scales,
            "interpolation_matrix": self.interpolation_matrix,
        }


    ### Model preparation

    @property
    def evaluate_func(self) -> BalmerEvaluate:
        return self.meta.get("evaluate_func", evaluate_exact)

    @evaluate_func.setter
    def evaluate_func(self, value: BalmerEvaluate) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> BalmerFitDeriv:
        return self.meta.get("fit_deriv_func", fit_deriv_exact_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: BalmerFitDeriv) -> None:
        self.meta["fit_deriv_func"] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop("fit_deriv_func", None)

    def _choose_evaluate_func(
        self, 
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self.evaluate_func = choose_evaluate_func(
            self.allow_interp_fitting,
            fixed or self.fixed,
        )

    def _choose_fit_deriv_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.allow_interp_fitting,
            fixed or self.fixed,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (2.0, 0.0)

    def _calculate_interpolation_matrices(self, x_out: NDArray[float64]) -> None:
        self._interpolation_matrices["interpolation_matrix"] = (
            create_interp_matrix.__wrapped__(
                self.continuum_template.x,
                x_out,
                left=0.0,
                right=0.0,
            )
        )

    @validate_call
    def rasterFit(
        self,
        x: FittableFloatVector,
        y: FittableFloatVector,
        dy: FittableFloatVector,
        *,
        inplace: bool = False,
    ) -> Self:
        """
        Performs a raster fit.

        Notes
        -----
        'ratio' parameter is held fixed at the current value.
        """
        assert not (self.flux.fixed and self.fwhm.fixed)

        if self.fwhm.fixed:
            fwhm = array([self.fwhm.value], dtype=float64)
            data = evaluate_exact(
                x,
                1.0, self.fwhm.value, self.ratio.value,
                **self._kwargs.update({'interpolation_matrix': None}),
                y=None,
            )[None, :]
        else:
            if array_equal(self.continuum_template.x, x):
                ctemp = self.continuum_template
                stemp = self.series_template
            else:
                ctemp = self.continuum_template.interpolate(x)
                stemp = self.series_template.interpolate(x)

            cdata = ctemp.data / ctemp.normalisation
            sdata = stemp.data / stemp.normalisation

            fwhm = ctemp.fwhm
            data = cdata + self.ratio.value * sdata

        if self.flux.fixed:
            flux_bounds = (self.flux.value, self.flux.value)
        else:
            flux_bounds = self.flux.bounds

        obj = self if inplace else self.copy()
        try:
            chi2s, fluxs = rasterise.__wrapped__(
                y,
                dy,
                fwhm,
                data,
                flux_bounds=flux_bounds,
                fwhm_bounds=self.fwhm.bounds,
            )
        except ValueError:
            return obj

        if (chi2s == 0).all():
            # ! Raise warning
            return obj

        chi2s[chi2s == 0] = inf
        idx = argmin(chi2s)

        if not self.flux.fixed:
            obj.flux.value = fluxs[idx]
        if not self.fwhm.fixed:
            obj.fwhm.value = apply_bounds(fwhm[idx], self.fwhm.bounds)

        return obj

    @validate_call
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
        wave = info.units.getWavelength(3000 * Unit("angstrom"))
        y_pl = model(wave)

        ctemp = self.continuum_template
        stemp = self.series_template

        y_ba = evaluate_exact(
            array([wave], dtype=float64),
            1.0,
            self.fwhm.value,
            self.ratio.value,
            template_x=ctemp.x,
            continuum_fwhm=ctemp.fwhm,
            continuum_data=ctemp.data / ctemp.normalisation,
            series_fwhm=stemp.fwhm,
            series_data=stemp.data / stemp.normalisation,
            sigma_res=ctemp.sigma_res,
            n_scales=3.0,
            interpolation_matrix=None,
            y=None,
        )[0]

        obj = self if inplace else self.copy()
        obj.flux.value = apply_bounds(a_qsfit * y_pl / y_ba, obj.flux.bounds)

        return obj

    ### Serialization

    def serialize(self, info: Info) -> dict[str, dict[str, Any]]:
        wave_unit = str(info.units.wavelength_unit)
        flux_unit = str(info.units.flux_unit)
        kms_unit = "km/s"
        data = {
            "name": self.name,
            "flux": serialize_parameter(self.flux, flux_unit),
            "fwhm": serialize_parameter(self.fwhm, kms_unit),
            "ratio": serialize_parameter(self.ratio, None),
            "edge": serialize_quantity(self.edge, wave_unit),
            "continuum_template": self.continuum_template.serialize(info),
            "series_template": self.series_template.serialize(info),
        }
        return {f"BalmerModel::{self.name}": data}

    @classmethod
    def deserialize(cls, data: dict[str, Any], name: str, info: Info) -> Self:
        wave_unit = str(info.units.wavelength_unit)
        flux_unit = str(info.units.flux_unit)
        kms_unit = "km/s"

        flux = deserialize_parameter(data["flux"], flux_unit)
        fwhm = deserialize_parameter(data["fwhm"], kms_unit)
        ratio = deserialize_parameter(data["ratio"], None)
        edge = deserialize_quantity(data["edge"], wave_unit)

        continuum_template = BalmerContinuumTemplate.deserialize(
            data["continuum_template"], 
            info,
        )
        series_template = BalmerSeriesTemplate.deserialize(
            data["series_template"], 
            info,
        )

        model = BalmerModel.create(
            flux["value"], fwhm["value"], ratio["value"], 
            edge=edge, 
            continuum_template=continuum_template, 
            series_template=series_template,
            info=info,
            allow_interp_fitting=info.convolution.allow_interp_fitting,
            flux_bounds=flux["bounds"],
            fwhm_bounds=fwhm["bounds"],
            ratio_bounds=ratio["bounds"],
            flux_fixed=flux["fixed"],
            fwhm_fixed=fwhm["fixed"],
            ratio_fixed=ratio["fixed"],
        )
        model.flux.tied = flux["tied"]
        model.fwhm.tied = fwhm["tied"]
        model.ratio.tied = ratio["tied"]

        model.name = name

        return model
