from typing import Self
from numpy import float64
from numpy.typing import NDArray
from astropy.modeling import Parameter

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from . import evaluation
from ...utils.basemodel import BaseModel

class BalmerSeriesModel(BaseModel):
    lumin = Parameter(default=1.0, min=0.0)
    fwhm = Parameter(default=5_000)

    def __init__(
        self,
        lumin: float,
        fwhm: float,
        *,
        sigma_res: float = None,
        edge: float = None,
        waves: NDArray[float64] = None,
        weights: NDArray[float64] = None,
        name: str = 'balmer_series',
        **kwargs,
    ): 
        assert sigma_res is not None, "sigma_res must be provided"
        assert edge is not None, "edge must be provided"
        assert waves is not None, "waves must be provided"
        assert weights is not None, "weights must be provided"
        
        super().__init__(
            lumin,
            fwhm,
            name=name,
            **kwargs,
        )
        self.sigma_res: float = sigma_res
        self.edge: float = edge
        self.waves: NDArray[float64] = waves
        self.weights: NDArray[float64] = weights / weights.sum()
    
    def evaluate(self, x, lumin, fwhm):
        return evaluation.evaluate(
            x,
            lumin, fwhm,
            sigma_res=self.sigma_res,
            waves=self.waves,
            weights=self.weights,
        )
    
    def fit_deriv(self, x, lumin, fwhm):
        return evaluation.fit_deriv(
            x,
            lumin, fwhm, 
            sigma_res=self.sigma_res,
            waves=self.waves,
            weights=self.weights,
            fixed=self.fixed,
        )
    
    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, BalmerSeriesModel):
            msg = f"Expected a BalmerSeriesModel instance, \
                got {type(value).__name__}"
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    def _ignore(self) -> None:
        """
        Ignore the Balmer series by nullifying its contribution.
        """
        assert not hasattr(self, '_old_specs'), "this model is already ignored"

        self._old_specs: dict[str, Parameter] = {
            'lumin': self.lumin.copy(),
            'fwhm': self.fwhm.copy(),
        }

        self.lumin.value  = 0.0
        self.lumin.fixed  = True
        self.lumin.bounds = None
        self.fwhm.fixed  = True

    def _restore(self) -> None:
        """
        Restore the Balmer series model to its previous state.
        """
        assert hasattr(self, '_old_specs'), "this model is not ignored"

        self.lumin = self._old_specs['lumin']
        self.fwhm = self._old_specs['fwhm']

        del self._old_specs