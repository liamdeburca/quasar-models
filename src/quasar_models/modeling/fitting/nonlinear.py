"""
Fitter class designed for 'SequentialModel' instances.
"""
__all__ = ["Fitter"]

from collections.abc import Callable
from dataclasses import field
from logging import getLogger
from time import perf_counter
from typing import Any, Literal, Union
from warnings import warn

from numpy import float64, ones
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import CompoundModel_
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_typing.scipy import OptimizeResult_
from scipy.optimize import least_squares

from ..base_model import BaseModel
from ..sequential_model import SequentialModel

logger = getLogger(__name__)


@dataclass
class Fitter:
    fit_info: OptimizeResult_ | None = field(default=None, init=False)
    sol: FloatVector | None = field(default=None, init=False)

    def __call__(
        self,
        model: Union[BaseModel, CompoundModel_[BaseModel], SequentialModel],
        x: FloatVector,
        y: FloatVector,
        *,
        dy: FloatVector | None = None,
        weights: FloatVector | None = None,
        where: BoolVector | None = None,
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
        calc_jac: bool = True,
    ) -> Union[BaseModel, CompoundModel_[BaseModel], None]:
        """
        Wrapper for `scipy.optimize.least_squares`.

        Parameters
        ----------
        model : BaseModel | CompoundModel_[BaseModel] | SequentialModel
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

        data: dict[str, FloatVector] = {
            "x": x,
            "y": y,
        }
        if (dy is None) and (weights is None):
            data["w"] = ones(x.size, dtype=float64)
        elif weights is not None:
            data["w"] = weights
        else:
            data["w"] = 1.0 / dy

        fun = lambda x: model.fun(x, data, where=where)
        if calc_jac:
            jac = lambda x: model.jac(x, data, where=where)
        else:
            jac = "2-point"

        t_start = perf_counter()
        res = least_squares(
            fun,
            model.x0,
            jac=jac,
            bounds=model.bounds,
            method=method,
            # Additional keyword arguments for least_squares
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        t_elapsed: float = (perf_counter() - t_start) * 1e3

        self.fit_info = OptimizeResult_.fromFitInfo(res)
        self.sol = model._get_full_parameter_array(self.fit_info.x)

        chi2n = self.fit_info.reduced_chi2
        n_lb = self.fit_info.n_lb
        n_ub = self.fit_info.n_ub
        nfev = self.fit_info.nfev
        status = self.fit_info.status
        message = self.fit_info.message

        if self.fit_info.success:
            msg = f"OptimizeResult [SUCCESS ({status})]: "
            log = logger.debug
        else:
            msg = f"OptimizeResult [FAILURE ({status})]: "
            log = logger.info
        log(msg + f"chi2/dof={chi2n:.1f}, {nfev=}/{max_nfev}, {n_lb=}, {n_ub=}, {message=}, {t_elapsed=:.1f} ms")

        if get_model:
            if inplace is None:
                msg = "'inplace' must be specified when 'get_model' is True."
                raise ValueError(msg)
            
            return model.get_final_model(
                params=self.fit_info.x,
                copy=not inplace,
            )
        elif inplace is not None:
            msg = "'inplace' should only be specified when 'get_model' is True."
            warn(msg, UserWarning)