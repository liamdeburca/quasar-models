"""
AstroPy compatible model: HostGalaxyModel.
"""
from logging import getLogger
from typing import Self
from numpy import isfinite, argmin
from astropy.modeling import Parameter

from quasar_typing.numpy import FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from .host_galaxy_template import HostGalaxyTemplate
from . import evaluation
from ..utils.astropy import apply_bounds
from ..utils.template import TemplateModel

logger = getLogger(__name__)

class HostGalaxyModel(TemplateModel):
    flux = Parameter(min=0)
    fwhm = Parameter(min=0)

    @validate_call
    def __init__(
        self,
        flux: float,
        fwhm: float,
        *,
        info: Info,
        template: HostGalaxyTemplate | None = None,
        allow_interp_fitting: bool = False,
        maxsize: int = 8,
        name: str = 'host_galaxy_model',
        **kwargs,
    ):
        if template is None:
            template = HostGalaxyTemplate.load_from_cache(name, info=info)
        else:
            if not template.info == info:
                msg = "The HostGalaxyTemplate's info instance does not match " \
                    "the HostGalaxyModel's info instance."
                raise ValueError(msg)
        
            name = template.name

        super().__init__(flux, fwhm, name=name, **kwargs)
        
        self.info: Info = info
        self.template: HostGalaxyTemplate = template
        self.allow_interp_fitting: bool = allow_interp_fitting

        self._initialise_cache(maxsize)

        # Update FWHM bounds using template
        self.fwhm.bounds = (template.fwhm[0], template.fwhm[-1])
        self.fwhm.value  = apply_bounds(self.fwhm.value, self.fwhm.bounds)

    @property
    def _perform_interp_fitting(self) -> bool:
        return self.allow_interp_fitting
    
    def evaluate(self, x, flux, fwhm):
        flux = float(flux)
        fwhm = float(fwhm)

        if self._perform_interp_fitting:
            return evaluation.evaluate_interp(
                x, flux, fwhm,
                host_galaxy_template=self.template,
                interpolation_matrix=self._get_interpolation_matrices(x),
            )
        return evaluation.evaluate(
            x, flux, fwhm,
            host_galaxy_template=self.template,
            interpolation_matrix=self._get_interpolation_matrices(x),
        )
    
    def fit_deriv(self, x, flux, fwhm):
        flux = float(flux)
        fwhm = float(fwhm)

        if self._perform_interp_fitting:
            return evaluation.fit_deriv_interp(
                x, flux, fwhm,
                host_galaxy_template=self.template,
                interpolation_matrix=self._get_interpolation_matrices(x),
            )
        return evaluation.fit_deriv(
            x, flux, fwhm,
            host_galaxy_template=self.template,
            interpolation_matrix=self._get_interpolation_matrices(x),
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
            x, y, dy,
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