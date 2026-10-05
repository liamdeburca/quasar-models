"""Power-law continuum model.

This module provides a small Astropy-style model wrapper around a
performance-oriented power-law implementation in
``quasar_models._core.modeling.powerlaw``. The public class
``PowerLawModel`` exposes parameters ``flux`` and ``alpha`` and helpers to
transform to/from log-space, evaluate the model, compute derivatives used by
fitting routines and construct models from linearised fits.

The implementation keeps changes minimal and delegates heavy numeric work to
the compiled core module.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any, ClassVar, Literal, Self

from astropy.modeling import Parameter
from numpy import exp, float64, log
from numpy.typing import NDArray
from pydantic_core import ValidationError
from quasar_typing.bounds import AstropyBounds
from quasar_typing.errors import OutsideBoundsError
from quasar_typing.numpy import FloatMatrix, FloatVector
from quasar_utils.decorators import validate_call
from quasar_utils.setup import Info

from quasar_models._core.modeling.powerlaw import (
    PowerLawEvaluate,
    PowerLawFitDeriv,
    choose_evaluate_func,
    choose_fit_deriv_func,
    evaluate,
    fit_deriv_all,
    inverse,
)
from quasar_models.modeling import BaseModel

from ..utils.astropy import apply_bounds
from ..utils.linear_regression import linreg
from ..utils.serialization import (
    deserialize_parameter,
    deserialize_quantity,
    serialize_parameter,
    serialize_quantity,
)

logger = getLogger(__name__)


@validate_call
def perform_linear_regression(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    *,
    x0: float,
    y0: float,
    flux_bounds: AstropyBounds,
    alpha_bounds: AstropyBounds,
) -> tuple[float, float]:
    """
    Performs linear regression. 

    Parameters
    ----------
    x : FloatVector
        The independent variable data.
    y : FloatVector
        The dependent variable data.
    dy : FloatVector
        The uncertainties in the dependent variable data.
    x0 : float
        The reference x value for the power law model.
    y0 : float
        The reference y value for the power law model.
    flux_bounds : AstropyBounds
        The bounds for the derived flux value.
    alpha_bounds : AstropyBounds
        The bounds for the derived alpha value.

    Returns
    -------
    flux : float
        The derived flux value from the linear regression.
    alpha : float
        The derived alpha value from the linear regression.

    Raises
    ------
    ValidationError
        If 'x', 'y', or 'dy' are not 1D numpy arrays.
        If 'flux_bounds' or 'alpha_bounds' are not valid bounds.
    OutsideBoundsError
        If the derived flux or alpha values are outside the specified bounds.
    """
    res: tuple[float, float] = linreg(x, y, dy)

    flux = y0 * exp(res[0])
    flux_lb, flux_ub = flux_bounds
    if flux_lb is not None and flux < flux_lb:
        msg = f"Derived flux value is below the lower bound: {flux=:.1f}<{flux_lb:.1f}"
        logger.info(msg)
        raise OutsideBoundsError(msg)
    if flux_ub is not None and flux_ub < flux:
        msg = f"Derived flux value is above the upper bound: {flux=:.1f}>{flux_ub:.1f}"
        logger.info(msg)
        raise OutsideBoundsError(msg)

    alpha = res[1]
    alpha_lb, alpha_ub = alpha_bounds
    if alpha_lb is not None and alpha < alpha_lb:
        msg = f"Derived alpha value is below the lower bound: {alpha=:.1f}<{alpha_lb:.1f}"
        logger.info(msg)
        raise OutsideBoundsError(msg)
    if alpha_ub is not None and alpha_ub < alpha:
        msg = f"Derived alpha value is above the upper bound: {alpha=:.1f}>{alpha_ub:.1f}"
        logger.info(msg)
        raise OutsideBoundsError(msg)

    return flux, alpha

class PowerLawModel(BaseModel):
    """Power-law continuum model wrapper.

    Parameters
    ----------
    flux, alpha : astropy.modeling.Parameter
        Model parameters. ``flux`` represents the normalisation at ``y0`` and
        ``alpha`` the power-law slope. Bounds are kept on the parameters and
        may be overridden on creation.

    Notes
    -----
    The class delegates evaluation and derivative calculations to compiled
    functions from ``quasar_models._core.modeling.powerlaw`` for performance.
    """
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

    model_type: ClassVar[Literal["pl"]] = "pl"

    @classmethod
    def create(
        cls,
        x0: float,
        y0: float,
        flux: float,
        alpha: float,
        name: str = "powerlaw",

        flux_bounds: AstropyBounds | None = None,
        alpha_bounds: AstropyBounds | None = None,
        flux_fixed: bool | None = None,
        alpha_fixed: bool | None = None,
    ) -> Self:
        model = PowerLawModel(
            flux,
            alpha,
            meta={"x0": x0, "y0": y0},
            name=name,
        )
        if flux_bounds is not None:
            model.flux.value = apply_bounds.__wrapped__(flux, flux_bounds)
            model.flux.bounds = flux_bounds
        if alpha_bounds is not None:
            model.alpha.value = apply_bounds.__wrapped__(alpha, alpha_bounds)
            model.alpha.bounds = alpha_bounds

        if flux_fixed is not None:
            model.flux.fixed = flux_fixed
        if alpha_fixed is not None:
            model.alpha.fixed = alpha_fixed

        return model

    @property
    def x0(self) -> float:
        return self.meta["x0"]

    @x0.setter
    def x0(self, value: float) -> None:
        self.meta["x0"] = value

    @property
    def y0(self) -> float:
        return self.meta["y0"]

    @y0.setter
    def y0(self, value: float) -> None:
        self.meta["y0"] = value

    def evaluate(self, x, flux, alpha, y=None):
        """Evaluate the power-law model.

        Parameters
        ----------
        x : array_like
            Independent variable (wavelength).
        flux, alpha : float or ndarray
            Model parameters. If arrays are passed, the first element will be
            used (this matches the small-wrapper behaviour used elsewhere in
            the package).
        y : array_like, optional
            If provided, the model output is added inplace to this array.

        Returns
        -------
        float or ndarray
            Model value(s) at ``x``.
        """
        return self.evaluate_func(
            x, 
            *self._transform_args_if_ndarray(flux, alpha), 
            **self.kwargs, 
            y=y,
        )

    def partial_deriv(self, x, flux, alpha, derivs=None):
        """Compute partial derivatives of the model.

        This returns an iterator (or generator) produced by the chosen
        ``fit_deriv_func`` implemented in the core module. The method enforces
        that scalar floats are provided for ``flux`` and ``alpha``.
        """
        return self.fit_deriv_func(
            x, 
            *self._transform_args_if_ndarray(flux, alpha),
            **self.kwargs,
            derivs=derivs,
        )

    def fit_deriv(self, x, flux, alpha, derivs=None):
        """Return derivatives as a list.

        Convenience wrapper around :meth:`partial_deriv` that materialises the
        returned iterator into a list (useful for callers that require
        concrete arrays).
        """
        return list(self.partial_deriv(x, flux, alpha, derivs=derivs))

    def inverse(self, y, flux, alpha):
        """Invert the model (solve for x given y).

        Delegates to the core ``inverse`` implementation.
        """
        return inverse(y, flux, alpha, **self.kwargs, x=None)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.meta.get("kwargs", {"x0": self.x0})

    @kwargs.setter
    def kwargs(self, value: dict[str, Any]) -> None:
        self.meta["kwargs"] = value

    @kwargs.deleter
    def kwargs(self) -> None:
        self.meta.pop("kwargs", None)

    def _set_kwargs(self) -> None:
        self.kwargs = {"x0": self.x0}

    ### Model preparation

    @property
    def evaluate_func(self) -> PowerLawEvaluate:
        return self.meta.get("evaluate_func", evaluate)

    @evaluate_func.setter
    def evaluate_func(self, value: PowerLawEvaluate) -> None:
        self.meta["evaluate_func"] = value

    @evaluate_func.deleter
    def evaluate_func(self) -> None:
        self.meta.pop("evaluate_func", None)

    @property
    def fit_deriv_func(self) -> PowerLawFitDeriv:
        return self.meta.get("fit_deriv_func", fit_deriv_all)

    @fit_deriv_func.setter
    def fit_deriv_func(self, value: PowerLawFitDeriv) -> None:
        self.meta["fit_deriv_func"] = value

    @fit_deriv_func.deleter
    def fit_deriv_func(self) -> None:
        self.meta.pop("fit_deriv_func", None)

    def _choose_evaluate_func(self) -> None:
        self.evaluate_func = choose_evaluate_func()

    def _choose_fit_deriv_func(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self.fit_deriv_func = choose_fit_deriv_func(fixed or self.fixed)

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
        """Compute residuals for data given the model.

        Parameters
        ----------
        x, y, dy : float or array-like
            Data and uncertainties. If ``log`` is True the inputs are expected
            to already be log-transformed (the helper transform functions will
            be used to convert values as required).
        log : bool
            If True, treat ``x`` and ``y`` as logarithmic values and apply the
            appropriate transforms before computing residuals.

        Returns
        -------
        float or ndarray
            Residuals (``(y - model)/dy``).
        """

        _x = self.transform_x.__wrapped__(self, x) if log else x
        _y = self.transform_y.__wrapped__(self, y) if log else y
        _dy = self.transform_dy.__wrapped__(self, y, dy) if log else dy

        _f = self.log(_x) if log else self(_x)

        return (_y - _f) / _dy

    @validate_call
    def log(self, x_log: float | FloatVector) -> float | FloatVector:
        """Evaluate the linearised (log) power-law.

        Returns log(flux/y0) + alpha * x_log which is useful for linear
        regression based estimation of parameters.
        """
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
        self, y: float | FloatVector, dy: float | FloatVector
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
        x_log: FloatVector,
        y_log: FloatVector,
        dy_log: FloatVector,
    ) -> Self:
        """Fit this model to data using linear regression.
        
        Creates a new PowerLawModel instance from a linear regression fit to the 
        provided data.

        Parameters
        ----------
        x_log : FloatVector
            Log-wavelength array.
        y_log : FloatVector
            Log-flux-density array.
        dy_log : FloatVector
            Log-flux-density uncertainty array.

        Raises
        ------
        OutsideBoundsError
            If the derived parameters from linear regression fall outside the 
            specified bounds.
        ValidationError
            If Pydantic validation fails.
        ValueError
            If any other unexpected error occurs during linear regression.
        """
        flux = self.flux.value
        alpha = self.alpha.value
        try:
            flux, alpha = perform_linear_regression.__wrapped__(
                x_log, y_log, dy_log,
                x0=self.x0,
                y0=self.y0,
                flux_bounds=self.flux.bounds,
                alpha_bounds=self.alpha.bounds
            )            
            msg = f"Linear regression successful: {flux=:.1f}, {alpha=:.1f}."
            logger.debug(msg)
        except OutsideBoundsError as e:
            msg = "Outside bounds error during linear regression (nonlinear optimisation recommended)"
            logger.info(msg)
            raise OutsideBoundsError(msg) from e
        except ValidationError as e:
            msg = "Validation error during linear regression"
            logger.info(msg)
            raise ValidationError(msg) from e
        except Exception as e:
            msg = f"Linear regression failed due to {type(e).__name__}: {e}"
            logger.info(msg)
            raise ValueError(msg) from e

        return PowerLawModel.create(
            self.x0,
            self.y0,
            flux, alpha, 
            name=self.name,
            flux_bounds=self.flux.bounds,
            alpha_bounds=self.alpha.bounds,
        )

    @validate_call
    def bootstrap(
        self,
        x: FloatVector,
        Y: FloatMatrix,
        dy: FloatVector,
    ) -> tuple[FloatVector, FloatVector]:
        assert x.size == Y.shape[-1]

        masks = Y > 0
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
        Returns function equivalent to f(...) = 1 / self.evaluate(...)
        """
        x0 = self.x0
        flux = self.flux.value
        alpha = self.alpha.value
        return lambda x: (1 / flux) * (x0 / x) ** alpha

    ### Serialization

    def serialize(self, info: Info) -> dict[str, dict[str, Any]]:
        wave_unit = str(info.units.wavelength_unit)
        flux_unit = str(info.units.flux_unit)
        data = {
            "x0": serialize_quantity(self.x0, wave_unit),
            "y0": serialize_quantity(self.y0, flux_unit),
            "flux": serialize_parameter(self.flux, flux_unit),
            "alpha": serialize_parameter(self.alpha, None),
        }
        return {f"PowerLawModel::{self.name}": data}

    @classmethod
    def deserialize(cls, data: dict[str, Any], name: str, info: Info) -> Self:
        wave_unit = str(info.units.wavelength_unit)
        flux_unit = str(info.units.flux_unit)

        x0: float = deserialize_quantity(data["x0"], wave_unit)
        y0: float = deserialize_quantity(data["y0"], flux_unit)
        flux = deserialize_parameter(data["flux"], flux_unit)
        alpha = deserialize_parameter(data["alpha"], None)

        out = PowerLawModel.create(
            x0, y0, 
            flux["value"], 
            alpha["value"],
            name=name,
            flux_bounds=flux["bounds"],
            alpha_bounds=alpha["bounds"],
            flux_fixed=flux["fixed"],
            alpha_fixed=alpha["fixed"],
        )
        out.flux.tied = flux["tied"]
        out.alpha.tied = alpha["tied"]

        return out
        
