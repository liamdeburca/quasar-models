"""
AstroPy compatible model: BalmerModel.
"""
from typing import Self, Literal, ClassVar
from numpy import array, float64, array_equal, unique, concatenate, nan, nanargmin, stack
from numpy.typing import NDArray
from astropy.units import Unit
from astropy.modeling import Parameter

from .continuum import BalmerContinuumTemplate
from .series import BalmerSeriesTemplate

from . import evaluation
from ..utils.template import TemplateModel
from ..utils.astropy import apply_bounds
from ..continuum import PowerLawModel

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

        if not continuum_template.temp == series_template.temp:
            msg = "The continuum ({}) and series ({}) template do not have " \
                "the same temperatures!".format(
                    continuum_template.temp, 
                    series_template.temp,
                )
            raise ValueError(msg)
        
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
    def same_xs(self) -> bool:
        return array_equal(self.continuum_template.x, self.series_template.x)

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

        if not self._interpolation_matrices:
            self._calculate_interpolation_matrices(x)

        if self._perform_interp_fitting:
            return evaluation.evaluate_interp(
                x, flux, fwhm, ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
                **self._interpolation_matrices,
            )
        return evaluation.evaluate(
            x, flux, fwhm, ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            **self._interpolation_matrices,
        )

    def fit_deriv(self, x, flux, fwhm, ratio):
        flux = float(flux)
        fwhm = float(fwhm)
        ratio = float(ratio)

        if not self._interpolation_matrices:
            self._calculate_interpolation_matrices(x)

        if self._perform_interp_fitting:
            return evaluation.fit_deriv_interp(
                x, flux, fwhm, ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
                fixed=self.fixed_dict,
                **self._interpolation_matrices,
            )
        return evaluation.fit_deriv(
            x, flux, fwhm, ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            fixed=self.fixed_dict,
            **self._interpolation_matrices,
        )
    
    def jac(self, x, flux, fwhm, ratio):
        return stack(self.fit_deriv(x, flux, fwhm, ratio), axis=0)
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (2.0, 0.0)
    
    def _calculate_interpolation_matrices(self, x_out: NDArray[float64]) -> None:
        """
        This method should be called previous to a fitting run, where the same
        interpolation matrix will be reused multiple times.
        """
        self._interpolation_matrices['continuum_interpolation_matrix'] \
            = create_interp_matrix.__wrapped__(
                self.continuum_template.x, x_out, left=0.0, right=0.0,
            )
        self._interpolation_matrices['series_interpolation_matrix'] \
            = create_interp_matrix.__wrapped__(
                self.series_template.x, x_out, left=0.0, right=0.0,
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
            data = evaluation.evaluate(
                x, 1.0, self.fwhm.value, self.ratio.value,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
            )[None,:]
        else:
            continuum_template = (
                self.continuum_template
                if array_equal(self.continuum_template.x, x) else
                self.continuum_template.interpolate(x, inplace=False)
            )
            series_template = (
                self.series_template
                if array_equal(self.series_template.x, x) else
                self.series_template.interpolate(x, inplace=False)
            )
            fwhm = continuum_template.fwhm
            data = continuum_template.data \
                + self.ratio.value * series_template.data

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
        y_ba = evaluation.evaluate(
            wave, 1.0, self.fwhm.value, self.ratio.value,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
        )

        obj = self if inplace else self.copy()
        obj.flux.value = apply_bounds(a_qsfit * y_pl / y_ba, obj.flux.bounds)

        return obj
