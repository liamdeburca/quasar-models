"""
AstroPy compatible model: IronModel.
"""
from logging import getLogger
from numpy import zeros_like, invert, nan, nanargmin, argmax, float64, array_equal, isnan
from numpy.typing import NDArray
from astropy.modeling import Parameter
from typing import Literal, Self
from scipy.sparse import csr_matrix

from pydantic import validate_call

from quasar_typing.numpy import FloatVector
from quasar_utils.interpolation import create_interp_matrix
from quasar_utils.raster import rasterise

from . import evaluation
from .iron_template import IronTemplate
from ..utils.basemodel import BaseModel
from ..utils.astropy import apply_bounds

logger = getLogger(__name__)

class IronModel(BaseModel):
    flux = Parameter(min=0)
    fwhm = Parameter(min=0)
    split = Parameter(default=1, min=0, fixed=True)
    left = Parameter(default=1, min=0, max=1, fixed=True)
    right = Parameter(default=1, min=0, max=1, fixed=True)

    @validate_call
    def __init__(
        self,
        template: IronTemplate,
        flux: float,
        fwhm: float,
        sigma_res: float,
        *,
        split: float = 1.0,
        left: float = 1.0,
        right: float = 1.0,
        scale: float = 1.0,
        allow_interp_fitting: bool = False,
        **kwargs,
    ):
        super().__init__(flux, fwhm, split, left, right, **kwargs)

        self.template: IronTemplate = template
        self.scale: float = scale
        self.sigma_res: float = sigma_res
        self.allow_interp_fitting: bool = allow_interp_fitting

        # Update FWHM bounds using template
        self.fwhm.bounds  = (template.fwhm[0], template.fwhm[-1])
        self.fwhm.value   = apply_bounds(self.fwhm.value, self.fwhm.bounds)
        
        # Update SPLIT bounds using template
        self.split.bounds = (template.x[0], template.x[-1])
        self.split.value  = apply_bounds(self.split.value, self.split.bounds)

        self._interpolation_cache: dict[
            tuple[int, int], 
            tuple[csr_matrix, NDArray[float64]],
        ] = {}
    
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
                interpolation_matrix=self._get_interpolation_matrix(x),
            )
        
        return evaluation.evaluate(
            x,
            flux, fwhm, split, left, right,
            sigma_res=self.sigma_res,
            scale=self.scale,
            template=self.template,
            interpolation_matrix=self._get_interpolation_matrix(x),
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
            sigma_res=self.sigma_res,
            scale=self.scale,
            template=self.template,
            interpolation_matrix=self._get_interpolation_matrix(x),
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
                interpolation_matrix=self._get_interpolation_matrix(x),
                fixed=self.fixed,
            )
        
        return evaluation.fit_deriv(
            x,
            flux, fwhm, split, left, right,
            sigma_res=self.sigma_res,
            scale=self.scale,
            template=self.template,
            interpolation_matrix=self._get_interpolation_matrix(x),
            fixed=self.fixed,
        )
    
    @property
    def model_type(self) -> Literal['fe']:
        """
        Returns a string-representation of the model type.
        """
        return 'fe'
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        return (1.0, self.template.x[argmax(self.template.data[0])])
    
    # Utilities

    def _get_interpolation_matrix(
        self,
        x_out: NDArray[float64],
    ) -> tuple[csr_matrix, NDArray[float64]]:
        """
        Retrieves an interpolation matrix from the cache which maps from the 
        IronTemplate's x-grid to the provided x_out grid. 
        """
        cache_key: tuple[int, int] = (x_out.size, hash(x_out.tobytes()))
        return self._interpolation_cache.get(cache_key, None)
    
    def _calculate_interpolation_matrix(
        self,
        x_out: NDArray[float64],
    ) -> tuple[csr_matrix, NDArray[float64]]:
        """
        Calculates an interpolation matrix which maps from the IronTemplate's
        x-grid to the provided x_out grid, and stores it in the cache.

        This method should be called previous to a fitting run, where the same
        interpolation matrix will be reused multiple times.
        """
        cache_key: tuple[int, int] = (x_out.size, hash(x_out.tobytes()))
        if cache_key not in self._interpolation_cache.keys():
            self._interpolation_cache[cache_key] = create_interp_matrix(
                self.template.x, x_out, left=0, right=0,
            )
        return self._interpolation_cache[cache_key]
    
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
        **PYDANTIC VALIDATED FUNCTION**

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
            if isnan(chi2s).all():
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

                if cond_l: flux = fluxs_left[idx]
                else:      flux = fluxs_right[idx]

                if cond_r: right = fluxs_right[idx] / flux
                else:      right = 1.0

                self.right.value = apply_bounds(right, self.right.bounds)
                self.right.fixed = False

            else:
                self.right.value = 1.0
                self.right.bounds = (min(self.right.bounds[0], 1.0), 1.0)
                self.right.fixed = True

                if cond_r: flux = fluxs_right[idx]
                else:      flux = fluxs_left[idx]

                if cond_l: left = fluxs_left[idx] / flux
                else:      left = 1.0

                self.left.value = apply_bounds(left, self.left.bounds)
                self.left.fixed = False

        return self