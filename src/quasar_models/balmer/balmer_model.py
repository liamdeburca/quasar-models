"""
AstroPy compatible model: BalmerModel.
"""
from typing import Self, Literal
from numpy import array, float64, array_equal, unique, concatenate, nan, nanargmin
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
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
    flux  = Parameter(default=1.0, min=0.0)
    fwhm  = Parameter(default=0,   min=0.0)
    ratio = Parameter(default=1.0, min=0.0)

    @validate_call
    def __init__(
        self,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        source: str | None = None,
        temp: float | None = None,
        tau: float | None = None,
        scale: float | None = None,
        dens: float | None = None,
        n_u_range: tuple[int, int] | None = None,
        info: Info = None,
        continuum_template: BalmerContinuumTemplate | None = None,
        series_template: BalmerSeriesTemplate | None = None,
        allow_interp_fitting: bool = False,
        maxsize: int = 8,
        name: str = 'balmer_pseudo_continuum',
        **kwargs,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **

        Notes
        -----
        The keyword arguments `edge`, `sigma_res`, `waves`, `weights`, and 
        `boltz` must be specified. Otherwise, a ValidationError is raised. 
        """
        if continuum_template is None:
            continuum_template = BalmerContinuumTemplate.load_from_cache(
                temp, tau, scale, info=info,
            )
        if not continuum_template.info == info:
            msg = "The BalmerContinuumTemplate's info instance does not " \
                "match the BalmerModel's info instance."
            raise ValueError(msg)

        if series_template is None:
            series_template = BalmerSeriesTemplate.load_from_cache(
                source, temp, dens, n_u_range, info=info,
            )
        elif not series_template.info == info:
            msg = "The BalmerSeriesTemplate's info instance does not match " \
                "the BalmerModel's info instance."
            raise ValueError(msg)

        if not continuum_template.temp == series_template.temp:
            msg = "The continuum ({}) and series ({}) template do not have " \
                "the same temperatures!".format(
                    info.units.getTemperature(continuum_template.temp), 
                    info.units.getTemperature(series_template.temp),
                )
            raise ValueError(msg)
        
        if not array_equal(continuum_template.fwhm, series_template.fwhm):
            fwhms = unique(
                concatenate([continuum_template.fwhm, series_template.fwhm]),
            )
            continuum_template.upsample(fwhms, inplace=True)
            series_template.upsample(fwhms, inplace=True)
            
        super().__init__(flux, fwhm, ratio, name=name, **kwargs)

        self.info: Info = info
        self.continuum_template: BalmerContinuumTemplate = continuum_template
        self.series_template: BalmerSeriesTemplate = series_template
        self.allow_interp_fitting: bool = allow_interp_fitting

        self._initialise_cache(maxsize)

        self.same_xs: bool = array_equal(
            self.continuum_template.x, 
            self.series_template.x,
        )

        # Update FWHM bounds using template
        self.fwhm.bounds = (continuum_template.fwhm[0], continuum_template.fwhm[-1])
        self.fwhm.value  = apply_bounds(self.fwhm.value, self.fwhm.bounds)

    @property
    def edge(self) -> float:
        return self.info.balmer.edge

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

        if self._perform_interp_fitting:
            return evaluation.evaluate_interp(
                x, flux, fwhm, ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
                **self._get_interpolation_matrices(x),
            )
        return evaluation.evaluate(
            x, flux, fwhm, ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            **self._get_interpolation_matrices(x),
        )

    def fit_deriv(self, x, flux, fwhm, ratio):
        flux = float(flux)
        fwhm = float(fwhm)
        ratio = float(ratio)

        if self._perform_interp_fitting:
            return evaluation.fit_deriv_interp(
                x, flux, fwhm, ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
                fixed=self.fixed,
                **self._get_interpolation_matrices(x),
            )
        return evaluation.fit_deriv(
            x, flux, fwhm, ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            fixed=self.fixed,
            **self._get_interpolation_matrices(x),
        )
    
    @property
    def model_type(self) -> Literal['ba']:
        """
        Returns a string-representation of the model type.
        """
        return 'ba'
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (2.0, 0.0)
    
    def _calculate_interpolation_matrices(
        self,
        x_out: NDArray[float64],
    ) -> dict[str, tuple[csr_matrix, NDArray[float64]]]:
        """
        This method should be called previous to a fitting run, where the same
        interpolation matrix will be reused multiple times.
        """
        cache_key: int = hash(x_out.tobytes())
        if cache_key not in self._interpolation_cache:
            interp_matrix_cont = create_interp_matrix(
                self.continuum_template.x, x_out, left=0, right=0,
            )
            interp_matrix_series = (
                interp_matrix_cont 
                if self.same_xs else 
                create_interp_matrix(self.series_template.x, x_out, left=0, right=0)
            )
            self._interpolation_cache[cache_key] = dict(
                continuum_interpolation_matrix=interp_matrix_cont, 
                series_interpolation_matrix=interp_matrix_series,
            )
        return self._interpolation_cache[cache_key]

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
