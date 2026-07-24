from logging import getLogger
from astropy.modeling import Parameter
from typing import Self, Callable, Literal, ClassVar
from numpy import log, exp, float64
from numpy.typing import NDArray

from pydantic_core import ValidationError

from quasar_models.modeling import BaseModel
from quasar_models.utils.astropy import apply_bounds
from quasar_models.utils.linear_regression import linreg
from quasar_models._core.modeling.powerlaw import (
    PowerLawEvaluate, PowerLawFitDeriv,
    choose_evaluate_func, choose_fit_deriv_func,
    evaluate, inverse,
    fit_deriv_all,
)

from quasar_utils.decorators import validate_call
from quasar_typing.numpy import FloatVector, FloatMatrix

logger = getLogger(__name__)

class PowerLawModel(BaseModel):
    flux = Parameter(
        default=1.0,
        bounds=(0.0, None),
        fixed=False,
    )
    alpha = Parameter(
        default=1.0,
        bounds=(-10.0, 10.0),
        fixed=False,
    )

    model_type: ClassVar[Literal['pl']] = 'pl'

    @classmethod
    def create(
        cls,
        x0: float,
        y0: float,
        flux: float,
        alpha: float,
        name: str = 'powerlaw',
    ) -> Self:
        return PowerLawModel(
            flux,
            alpha,
            meta={'x0': x0, 'y0': y0},
            name=name,
        )
    
    @property
    def x0(self) -> float:
        return self.meta['x0']
    
    @x0.setter
    def x0(self, value: float) -> None:
        self.meta['x0'] = value
    
    @property
    def y0(self) -> float:
        return self.meta['y0']

    @y0.setter
    def y0(self, value: float) -> None:
        self.meta['y0'] = value

    def evaluate(self, x, flux, alpha):
        return self.evaluate_func(x, flux, alpha, **self._kwargs, y=None)
    
    def jac(self, x, flux, alpha):
        return self.fit_deriv_func(x, flux, alpha, **self._kwargs, derivs=None)

    def fit_deriv(self, x, flux, alpha):
        return list(self.jac(x, flux, alpha))

    def inverse(self, y, flux, alpha):
        return inverse(y, flux, alpha, **self._kwargs, x=None)

    ### Model preparation

    @property
    def evaluate_func(self) -> PowerLawEvaluate:
        return self.meta.get('evaluate_func', evaluate)
    
    @evaluate_func.setter
    def evaluate_func(self, value: PowerLawEvaluate) -> None:
        self.meta['evaluate_func'] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop('evaluate_func', None)
    
    @property
    def fit_deriv_func(self) -> PowerLawFitDeriv:
        return self.meta.get('fit_deriv_func', fit_deriv_all)
    
    @fit_deriv_func.setter
    def fit_deriv_func(self, value: PowerLawFitDeriv) -> None:
        self.meta['fit_deriv_func'] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop('fit_deriv_func', None)

    def _choose_evaluate_func(self) -> None:
        self.evaluate_func = choose_evaluate_func()
    
    def _choose_fit_deriv_func(self) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(
            self.fixed_dict or self.fixed,
        )

    @property
    def _kwargs(self) -> dict:
        return {
            'x0': self.x0,
        }
    
    # Utilities
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Returns a key used for sorting submodels of compound models. The key is
        a tuple. 
        """
        return (0.0, 0.0)
    
    ### Custom utility methods
    @validate_call
    def getResiduals(
        self,
        x: float | FloatVector,
        y: float | FloatVector,
        dy: float | FloatVector,
        log: bool = False,
    ) -> float | FloatVector:
        
        _x = self.transform_x  .__wrapped__(self, x)     if log else x
        _y = self.transform_y  .__wrapped__(self, y)     if log else y
        _dy = self.transform_dy.__wrapped__(self, y, dy) if log else dy
        
        _f = self.log(_x) if log else self(_x)
        
        return (_y - _f) / _dy
    
    @validate_call
    def log(
        self,
        x_log: float | FloatVector
    ) -> float | FloatVector:
        return log(self.flux.value / self.y0) + self.alpha.value * x_log
    
    @validate_call
    def transform_x(
        self,
        x: float | FloatVector,
    ) -> float | FloatVector:
        return log(x) - log(self.x0)
    
    @validate_call
    def inv_transform_x(
        self,
        x_log: float | FloatVector,
    ) -> float | FloatVector:
        return self.x0 * exp(x_log)
    
    @validate_call
    def transform_y(
        self,
        y: float | FloatVector,
    ) -> float | FloatVector:
        return log(y) - log(self.y0)
    
    @validate_call
    def inv_transform_y(
        self,
        y_log: float | FloatVector,
    ) -> float | FloatVector:
        return self.y0 * exp(y_log)

    @validate_call
    def transform_dy(
        self,
        y: float | FloatVector,
        dy: float | FloatVector
    ) -> float | FloatVector:
        return dy / y
    
    @validate_call
    def inv_transform_dy(
        self,
        y_log: float | FloatVector,
        dy_log: float | FloatVector,
    ) -> float | FloatVector:
        return dy_log * self.inv_transform_y.__wrapped__(self, y_log)

    @validate_call
    def from_linear_fit(
        self,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
    ) -> Self:
        flux = self.flux.value
        alpha = self.alpha.value
        try:
            a, b = linreg(
                self.transform_x.__wrapped__(self, x),
                self.transform_y.__wrapped__(self, y),
                self.transform_dy.__wrapped__(self, y, dy),
            )
            flux = apply_bounds(self.y0 * exp(a), self.flux.bounds)
            alpha = apply_bounds(b, self.alpha.bounds)
            msg = "Linear regression successful: " \
                f"{a=:.3e}, {b=:.3f} | {flux=:.3e}, {alpha=:.3f}."
            logger.debug(msg)
        except ValidationError as e:
            msg = f"Linear regression failed due to validation error: {e}"
            logger.warning(msg)
        except ValueError as e:
            msg = f"Linear regression failed due to value error: {e}"
            logger.warning(msg)
        except Exception as e:
            msg = f"Linear regression failed due to unexpected error: {e}"
            logger.warning(msg)
            
        model = PowerLawModel.create(
            self.x0, 
            self.y0, 
            flux, 
            alpha, 
            name=self.name,
        )
        model.flux.bounds = self.flux.bounds
        model.alpha.bounds = self.alpha.bounds

        return model
    
    @validate_call
    def bootstrap(
        self,
        x: FloatVector,
        Y: FloatMatrix,
        dy: FloatVector,
    ) -> tuple[FloatVector, FloatVector]:
        assert x.size == Y.shape[-1]

        masks = (Y > 0)
        if not (valid_rows := (masks.sum(axis=-1) >= 2)).all():
            Y = Y[valid_rows]
            masks = masks[valid_rows]

        a: list[float] = []
        b: list[float] = []
        for mask, y in zip(masks, Y):
            res = linreg(
                self.transform_x.__wrapped__(self, x[mask]),
                self.transform_y.__wrapped__(self, y[mask]),
                self.transform_dy.__wrapped__(self, y[mask], dy[mask]),
            )
            a.append(res[0])
            b.append(res[1])

        flux = apply_bounds(self.y0 * exp(a), self.flux.bounds)
        alpha = apply_bounds(b, self.alpha.bounds)

        return flux, alpha
    
    ### Convenience function(s) for bootstrapping 

    def _get_flipped_func(
        self,
    ) -> Callable[[float | NDArray[float64]], float | NDArray[float64]]:
        """
        Returns function equavalent to f(...) = 1 / self.evaluate(...)
        """
        x0 = self.x0
        flux = self.flux.value
        alpha = self.alpha.value
        return lambda x: (1 / flux) * (x0 / x)**alpha