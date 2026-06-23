"""
AstroPy compatible model: IronModel.
"""
from logging import getLogger
from numpy import zeros_like, invert, nan, nanargmin, stack, float64, array_equal, isfinite
from astropy.modeling import Parameter
from typing import Literal, Self, ClassVar

from quasar_typing.numpy import FloatVector
from quasar_utils.setup import Info
from quasar_utils.decorators import validate_call
from quasar_utils.raster import rasterise

from . import evaluation
from .iron_template import IronTemplate
from ..utils.template import TemplateModel
from ..utils.astropy import apply_bounds

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

    model_type: ClassVar[Literal['fe']] = 'fe'

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
        name: str | Literal['vw2001', 'v2003', 'bw'] | None = None,
    ) -> Self:
        if template is None:
            if info is None:
                msg = "'info' cannot be None if 'template' is not provided."
                raise ValueError(msg)
            if name is None:
                msg = "'name' cannot be None if 'template' is not provided."
                raise ValueError(msg)
            
            template = IronTemplate.load_from_cache(name=name, info=info)

        model = IronModel(
            flux, fwhm,
            split=split, left=left, right=right,
            name=template.name,
            meta={
                'template': template,
                'scale': scale,
                'allow_interp_fitting': allow_interp_fitting,
                '_interpolation_matrices': {},
            }
        )
        model.split.bounds = (
            template.x[0],
            template.x[-1],
        )
        model.split.value = apply_bounds.__wrapped__(
            model.split.value,
            model.split.bounds,
        )
        return model
    
    @property
    def template(self) -> IronTemplate: return self.meta['template']

    @template.setter
    def template(self, value: IronTemplate) -> None:
        self.meta['template'] = value

    @property
    def scale(self) -> float: return self.meta['scale']

    @scale.setter
    def scale(self, value: float) -> None:
        self.meta['scale'] = value
    
    @property
    def _perform_interp_fitting(self) -> bool:
        return self.allow_interp_fitting \
            and (self.left.fixed and self.left.value == 1.0) \
            and (self.right.fixed and self.right.value == 1.0)
    
    def evaluate(self, x, flux, fwhm, split, left, right):
        flux = float(flux)
        fwhm = float(fwhm)
        split = float(split)
        left = float(left)
        right = float(right)
        
        if self._perform_interp_fitting:
            return evaluation.evaluate_interp(
                x,
                flux, fwhm,
                template=self.template,
                **self._interpolation_matrices,
            )
        
        return evaluation.evaluate(
            x,
            flux, fwhm, split, left, right,
            template=self.template,
            **self._interpolation_matrices,
        )
    
    def evaluate_sparse(self, x, flux, fwhm, split, left, right):
        flux = float(flux)
        fwhm = float(fwhm)
        split = float(split)
        left = float(left)
        right = float(right)
        
        return evaluation.evaluate_sparse(
            x,
            flux, fwhm, split, left, right,
            template=self.template,
            **self._interpolation_matrices,
        )
    
    def fit_deriv(self, x, flux, fwhm, split, left, right):
        flux = float(flux)
        fwhm = float(fwhm)
        split = float(split)
        left = float(left)
        right = float(right)

        if self._perform_interp_fitting:
            return evaluation.fit_deriv_interp(
                x,
                flux, fwhm,
                template=self.template,
                fixed=self.fixed_dict,
                **self._interpolation_matrices,
            )
        
        return evaluation.fit_deriv(
            x,
            flux, fwhm, split, left, right,
            template=self.template,
            fixed=self.fixed_dict,
            **self._interpolation_matrices,
        )
    
    def jac(self, x, flux, fwhm, split, left, right):
        return stack(self.fit_deriv(x, flux, fwhm, split, left, right), axis=0)
            
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
        bias: Literal['left', 'right'] = 'right',
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

        template = (
            self.template \
            if array_equal(x, self.template.x) else 
            self.template.interpolate(x, inplace=False)
        )

        # Perform a single raster fit
        if self.left.fixed and self.right.fixed:
            data = template.data * template._get_split_weight(
                x, 
                self.split.value, self.left.value, self.right.value, 
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

            idx: int = nanargmin(chi2s)
            self.flux.value = fluxs[idx]
            self.fwhm.value = template.fwhm[idx]

        # Perform separate raster fits for left and right
        else:
            is_left  = (self.template.x < self.split.value)
            is_right = invert(is_left)

            mask_left  = (x < self.split.value)
            mask_right = invert(mask_left)

            chi2s = zeros_like(self.template.fwhm, dtype=float64)
            if cond_l := (is_left.any() and (mask_left.sum() >= 2)):
                chi2s_left, fluxs_left = rasterise.__wrapped__(
                    y[mask_left], 
                    dy[mask_left], 
                    self.template.fwhm, 
                    template.data[:,mask_left],
                    flux_bounds = self.flux.bounds,
                    fwhm_bounds = self.fwhm.bounds,
                )
                chi2s += chi2s_left
            if cond_r := (is_right.any() and (mask_right.sum() >= 2)):
                chi2s_right, fluxs_right = rasterise.__wrapped__(
                    y[mask_right], 
                    dy[mask_right], 
                    self.template.fwhm, 
                    template.data[:,mask_right],
                    flux_bounds = self.flux.bounds,
                    fwhm_bounds = self.fwhm.bounds,
                )
                chi2s += chi2s_right

            chi2s[chi2s == 0] = nan
            idx: int = nanargmin(chi2s)

            self.flux.value = fluxs[idx]
            self.fwhm.value = apply_bounds(template.fwhm[idx], self.fwhm.bounds)

            if bias == 'left':
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