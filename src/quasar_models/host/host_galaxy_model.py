"""
AstroPy compatible model: HostGalaxyModel.
"""
from logging import getLogger
from typing import Self, Literal, ClassVar
from numpy import isfinite, argmin, stack
from astropy.modeling import Parameter

from quasar_typing.numpy import FloatVector

from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from .io import convert_params_to_name

from .host_galaxy_template import HostGalaxyTemplate
from . import evaluation
from ..utils.astropy import apply_bounds
from ..utils.template import TemplateModel

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

    model_type: ClassVar[Literal['hg']] = 'hg'

    @classmethod
    def create(
        cls,
        flux: float,
        fwhm: float,
        *,
        info: Info | None = None,
        template: HostGalaxyTemplate | None = None,
        allow_interp_fitting: bool = False,
        name: Literal['bc2003'] | None = None,
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
            flux, fwhm,
            name=convert_params_to_name(template.name, template.age),
            meta={
                'template': template,
                'allow_interp_fitting': allow_interp_fitting,
                '_interpolation_matrices': {},
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
        return self.meta['template']

    @template.setter
    def template(self, value: HostGalaxyTemplate) -> None:
        self.meta['template'] = value

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (3.0, self.template.x_norm)

    ### 
    
    def evaluate(self, x, flux, fwhm):
        flux = float(flux)
        fwhm = float(fwhm)

        if self._perform_interp_fitting:
            return evaluation.evaluate_interp(
                x, flux, fwhm,
                host_galaxy_template=self.template,
                **self._interpolation_matrices,
            )
        return evaluation.evaluate(
            x, flux, fwhm,
            host_galaxy_template=self.template,
            **self._interpolation_matrices,
        )
    
    def fit_deriv(self, x, flux, fwhm):
        flux = float(flux)
        fwhm = float(fwhm)

        if self._perform_interp_fitting:
            return evaluation.fit_deriv_interp(
                x, flux, fwhm,
                host_galaxy_template=self.template,
                fixed=self.fixed_dict,
                **self._interpolation_matrices,
            )
        return evaluation.fit_deriv(
            x, flux, fwhm,
            host_galaxy_template=self.template,
            **self._interpolation_matrices,
            fixed=self.fixed_dict,
        )
    
    def jac(self, x, flux, fwhm):
        return stack(self.fit_deriv(x, flux, fwhm), axis=0)
    
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