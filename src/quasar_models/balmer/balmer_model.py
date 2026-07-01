"""
AstroPy compatible model: BalmerModel.
"""
from typing import Self, Literal, ClassVar
from numpy import array, float64, array_equal, unique, concatenate, nan, nanargmin
from numpy.typing import NDArray
from astropy.units import Unit
from astropy.modeling import Parameter

from .continuum import BalmerContinuumTemplate
from .series import BalmerSeriesTemplate

from ..utils.template import TemplateModel
from ..utils.astropy import apply_bounds
from ..continuum import PowerLawModel

from quasar_core.modelling.balmer import (
    BalmerEvaluate, BalmerFitDeriv,
    choose_evaluate_func, choose_fit_deriv_func,
    evaluate_exact, fit_deriv_exact_all,
)

from quasar_utils.setup import Info
from quasar_utils.raster import rasterise
from quasar_utils.decorators import validate_call
from quasar_utils.interpolation import create_interp_matrix

from quasar_typing.numpy import FittableFloatVector, FloatVector

class BalmerModel(TemplateModel):
    flux = Parameter(
        default=1.0, 
        min=0.0,
    )
    fwhm = Parameter(
        default=0,   
        min=0.0,
    )
    ratio = Parameter(
        default=1.0, 
        min=0.0,
    )

    model_type: ClassVar[Literal['ba']] = 'ba'

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
        name: str | Literal['SH1995'] | None = None,
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
            msg = "The continuum and series templates do not have "\
                "the same x arrays!"
            raise ValueError(msg)

        if not continuum_template.temp == series_template.temp:
            msg = "The continuum ({}) and series ({}) template do not have " \
                "the same temperatures!".format(
                    continuum_template.temp, 
                    series_template.temp,
                )
            raise ValueError(msg)
        
        if n_scales is not None:
            continuum_template.n_scales = n_scales
            series_template.n_scales = n_scales
        elif not continuum_template.n_scales == series_template.n_scales:
            msg = "The continuum ({}) and series ({}) template do not have " \
                "the same 'n_scales' values! Setting according to maximum "\
                "value.".format(
                    continuum_template.n_scales, 
                    series_template.n_scales,
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
            name=name or 'balmer',
            meta={
                'continuum_template': continuum_template,
                'series_template': series_template,
                'allow_interp_fitting': allow_interp_fitting,
                'edge': edge,
                '_interpolation_matrices': {},
            }
        )
        model.fwhm.bounds = (continuum_template.fwhm[0], continuum_template.fwhm[-1])
        model.fwhm.value = apply_bounds.__wrapped__(
            model.fwhm.value, 
            model.fwhm.bounds,
        )

        return model
    
    @property
    def continuum_template(self) -> BalmerContinuumTemplate:
        return self.meta['continuum_template']
    
    @continuum_template.setter
    def continuum_template(self, value: BalmerContinuumTemplate) -> None:
        self.meta['continuum_template'] = value

    @property
    def series_template(self) -> BalmerSeriesTemplate:
        return self.meta['series_template']
    
    @series_template.setter
    def series_template(self, value: BalmerSeriesTemplate) -> None:
        self.meta['series_template'] = value

    @property 
    def n_scales(self) -> float:
        return self.continuum_template.n_scales
    
    @n_scales.setter
    def n_scales(self, value: float) -> None:
        self.continuum_template.n_scales = value
        self.series_template.n_scales = value

    @property
    def edge(self) -> float:
        return self.meta['edge']
    
    @edge.setter
    def edge(self, value: float) -> None:
        self.meta['edge'] = value

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
    
    def evaluate(self, x, flux, fwhm, ratio):
        flux = float(flux)
        fwhm = float(fwhm)
        ratio = float(ratio)
        return self.evaluate_func(
            x,
            flux, fwhm, ratio,
            **self._kwargs,
            y=None,
        )

    def jac(self, x, flux, fwhm, ratio):
        flux = float(flux)
        fwhm = float(fwhm)
        ratio = float(ratio)
        return self.fit_deriv_func(
            x, 
            flux, fwhm, ratio, 
            **self._kwargs, derivs=None,
        )
    
    def fit_deriv(self, x, flux, fwhm, ratio):
        return list(self.jac(x, flux, fwhm, ratio))
    
    ### Model preparation

    @property
    def evaluate_func(self) -> BalmerEvaluate:
        return self.meta.get('evaluate_func', evaluate_exact)
    
    @evaluate_func.setter
    def evaluate_func(self, value: BalmerEvaluate) -> None:
        self.meta['evaluate_func'] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop('evaluate_func', None)
    
    @property
    def fit_deriv_func(self) -> BalmerFitDeriv:
        return self.meta.get('fit_deriv_func', fit_deriv_exact_all)
    
    @fit_deriv_func.setter
    def fit_deriv_func(self, value: BalmerFitDeriv) -> None:
        self.meta['fit_deriv_func'] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop('fit_deriv_func', None)

    def _choose_evaluate_func(self) -> None:
        self.evaluate_func = choose_evaluate_func(
            self.allow_interp_fitting,
        )
    
    def _choose_fit_deriv_func(self) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.allow_interp_fitting,
            self.fixed_dict or self.fixed,
        )

    def _prepare_model(self, x_out: FloatVector, *args) -> None:
        self.fixed_dict = {
            'flux': self.flux.fixed,
            'fwhm': self.fwhm.fixed,
            'ratio': self.ratio.fixed,
        }
        self._choose_evaluate_func()
        self._choose_fit_deriv_func()
        self._calculate_interpolation_matrices(x_out)

    @property
    def _kwargs(self) -> dict:
        c = self.continuum_template
        s = self.series_template
        out = {
            'template_x': c.x,
            'continuum_fwhm': c.fwhm,
            'continuum_data': c.data / c.normalisation,
            'series_fwhm': s.fwhm,
            'series_data': s.data / s.normalisation,
            'sigma_res': c.sigma_res,
            'n_scales': c.n_scales,
        }
        out.update(self._interpolation_matrices)
        return out
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (2.0, 0.0)
    
    def _calculate_interpolation_matrices(self, x_out: NDArray[float64]) -> None:
        self._interpolation_matrices['interpolation_matrix'] = \
            create_interp_matrix.__wrapped__(
                self.continuum_template.x, x_out,
                left=0.0, right=0.0,
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
                template_x=self.continuum_template.x,
                continuum_fwhm=self.continuum_template.fwhm,
                continuum_data=self.continuum_template.data / self.continuum_template.normalisation,
                series_fwhm=self.series_template.fwhm,
                series_data=self.series_template.data / self.series_template.normalisation,
                sigma_res=self.continuum_template.sigma_res,
                n_scales=self.n_scales,
                interpolation_matrix=None,
                y=None,
            )[None,:]
        else:
            if array_equal(self.continuum_template.x, x):
                ctemp = self.continuum_template
                stemp = self.series_template
            else:
                ctemp = self.continuum_template.interpolate(x, inplace=False)
                stemp = self.series_template.interpolate(x, inplace=False)

            fwhm = ctemp.fwhm
            data = ctemp.data + self.ratio.value * stemp.data

        if self.flux.fixed:
            flux_bounds = (self.flux.value, self.flux.value)
        else:
            flux_bounds = self.flux.bounds

        chi2s, fluxs = rasterise.__wrapped__(
            y, dy,
            fwhm,
            data,
            flux_bounds=flux_bounds,
            fwhm_bounds=self.fwhm.bounds,
        )
        
        obj = self if inplace else self.copy()
        if (chi2s == 0).all():
            #! Raise warning
            return obj

        chi2s[chi2s == 0] = nan
        idx = nanargmin(chi2s)

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
        wave = info.units.getWavelength(3000 * Unit('angstrom'))
        y_pl = model(wave)

        y_ba = evaluate_exact(
            array([wave], dtype=float64),
            1.0, self.fwhm.value, self.ratio.value,
            template_x=self.continuum_template.x,
            continuum_fwhm=self.continuum_template.fwhm,
            continuum_data=self.continuum_template.data / self.continuum_template.normalisation,
            series_fwhm=self.series_template.fwhm,
            series_data=self.series_template.data / self.series_template.normalisation,
            sigma_res=self.continuum_template.sigma_res,
            n_scales=3.0,
            interpolation_matrix=None,
            y=None,
        )[0]

        obj = self if inplace else self.copy()
        obj.flux.value = apply_bounds(a_qsfit * y_pl / y_ba, obj.flux.bounds)

        return obj
