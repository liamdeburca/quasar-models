"""
Fitter class designed for 'SequentialModel' instances.
"""
from typing import Literal, ClassVar, Union, Any, Callable
from dataclasses import dataclass, field
from astropy.modeling import CompoundModel
from numpy import float64, ones
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from quasar_models.modeling import BaseModel, SequentialModel

@dataclass
class Fitter:
    calc_uncertainties: bool = field(default=False)

    fit_info: OptimizeResult | None = field(default=None, init=False)
    sol: NDArray[float64] | None = field(default=None, init=False)

    method: ClassVar[Literal['lm', 'trf', 'dogbox']]

    def __call__(
        self,
        model: Union[BaseModel, CompoundModel, SequentialModel],
        x: NDArray[float64],
        y: NDArray[float64],
        *,
        dy: NDArray[float64] | None = None,
        weights: NDArray[float64] | None = None,
        
        get_model: bool = False,
        inplace: bool = False,

        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        x_scale: Any | None = None,
        loss: str | Callable = 'linear',
        f_scale: float = 1.0,
        max_nfev: int | None = None,
    ) -> Union[BaseModel, SequentialModel, None]:
        if not isinstance(model, SequentialModel):
            model = SequentialModel(model)

        data: dict[str, NDArray[float64]] = {
            'x': x,
            'y': y,
        }
        if (dy is None) and (weights is None):
            data['w'] = ones(x.size, dtype=float64)
        elif weights is not None:
            data['w'] = weights
        else:
            data['w'] = 1.0 / dy

        self.fit_info = least_squares(
            model.fun,
            model.x0,
            jac=model.jac,
            bounds=model.bounds,
            method=self.method,
            args=(data,),

            # Additional keyword arguments for least_squares
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev
        )
        self.sol = model._get_full_parameter_array(self.fit_info.x)

        if get_model:
            return model.get_final_model(
                params=self.fit_info.x,
                copy=not inplace,
            )

@dataclass
class LMLSQFitter(Fitter):
    method: ClassVar[Literal['lm']] = 'lm'

@dataclass
class TRFLSQFitter(Fitter):
    method: ClassVar[Literal['trf']] = 'trf'

@dataclass
class DogBoxLSQFitter(Fitter):
    method: ClassVar[Literal['dogbox']] = 'dogbox'