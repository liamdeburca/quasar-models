"""
Fitter class designed for 'SequentialModel' instances.
"""
__all__ = ["Fitter"]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union
from warnings import warn

from astropy.modeling import CompoundModel
from numpy import float64, ones
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from quasar_models.modeling import BaseModel, SequentialModel


@dataclass
class Fitter:
    fit_info: OptimizeResult | None = field(default=None, init=False)
    sol: NDArray[float64] | None = field(default=None, init=False)

    def __call__(
        self,
        model: Union[BaseModel, CompoundModel, SequentialModel],
        x: NDArray[float64],
        y: NDArray[float64],
        *,
        dy: NDArray[float64] | None = None,
        weights: NDArray[float64] | None = None,
        get_model: bool = False,
        inplace: bool | None = None,
        method: Literal["lm", "trf", "dogbox"] = "trf",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        x_scale: Any | None = None,
        loss: str | Callable = "linear",
        f_scale: float = 1.0,
        max_nfev: int | None = None,
    ) -> Union[BaseModel, CompoundModel] | None:
        """
        Wrapper for `scipy.optimize.least_squares`.

        Parameters
        ----------
        model : BaseModel | CompoundModel | SequentialModel
            The model to fit to the data. If this model is not an instance of 
            `SequentialModel`, an instance will be created.
        x : NDArray[float64]
            The independent variable data.
        y : NDArray[float64]
            The data to fit the model to.
        dy : NDArray[float64] | None, optional
            The uncertainties in the data. Only used if the `weights` argument 
            is not provided. If both `dy` and `weights` are None, no weights are
            applied.
        weights : NDArray[float64] | None, optional
            The weights to apply to the data. If both `dy` and `weights` are 
            None, no weights are applied.
        get_model : bool, optional
            If True and `inplace` is not None, the fitted Astropy model is returned.
        inplace : bool | None, optional
            If True, the fitted Astropy model is returned as a copy. If False, 
            the fitted Astropy model is returned in place. A boolean value must 
            be provided if `get_model` is True. If a boolean value is provided, 
            but `get_model` is False, a warning is raised.
        ftol : float, optional
            See `scipy.optimize.least_squares` documentation.
        xtol : float, optional
            See `scipy.optimize.least_squares` documentation.
        gtol : float, optional
            See `scipy.optimize.least_squares` documentation.
        x_scale : Any | None, optional
            See `scipy.optimize.least_squares` documentation.
        loss : str | Callable, optional
            See `scipy.optimize.least_squares` documentation.
        f_scale : float, optional
            See `scipy.optimize.least_squares` documentation.
        max_nfev : int | None, optional
            See `scipy.optimize.least_squares` documentation.
        """
        if not isinstance(model, SequentialModel):
            model = SequentialModel(model)

        data: dict[str, NDArray[float64]] = {
            "x": x,
            "y": y,
        }
        if (dy is None) and (weights is None):
            data["w"] = ones(x.size, dtype=float64)
        elif weights is not None:
            data["w"] = weights
        else:
            data["w"] = 1.0 / dy

        self.fit_info = least_squares(
            model.fun,
            model.x0,
            jac=model.jac,
            bounds=model.bounds,
            method=method,
            args=(data,),
            # Additional keyword arguments for least_squares
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        self.sol = model._get_full_parameter_array(self.fit_info.x)

        if get_model:
            if not isinstance(inplace, bool):
                msg = "'inplace' must be specified when 'get_model' is True."
                raise ValueError(msg)
            
            return model.get_final_model(
                params=self.fit_info.x,
                copy=not inplace,
            )
        elif isinstance(inplace, bool):
            msg = "'inplace' should only be specified when 'get_model' is True."
            warn(msg, UserWarning)