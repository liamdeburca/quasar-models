from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

from astropy.modeling import CompoundModel, Parameter
from numpy import (
    array,
    einsum,
    empty,
    float64,
    fromiter,
    int_,
    isfinite,
    zeros,
)
from numpy.typing import NDArray

from ..utils.astropy import apply_bounds
from .base_model import BaseModel
from .utils import (
    LinearTie,
    Param,
    get_param_type,
)


def _validate_compound_model(model: CompoundModel) -> None:
    """
    Validates that the given compound model is a valid sequential model, i.e. it
    is constructed using only the '+' operator. Other operators are not yet
    allowed.
    """
    m = model
    while True:
        if not isinstance(m, CompoundModel):
            break

        if m.op != "+":
            msg = "Compound model must be constructed using only the '+' operator."
            raise ValueError(msg)

        m = m.left


def _validate_initial_values(params: Iterable[Param | Parameter]) -> None:
    for param in params:
        if not isfinite(val := param.value):
            msg = f"Parameter '{param.name}' has non-finite value: {val}."
            raise ValueError(msg)

        if not param.fixed:
            lb, ub = param.bounds

            if (lb is not None) and val < lb:
                msg = f"Parameter '{param.name}' has value {val} below lower bound {lb}."
                raise ValueError(msg)
            
            if (ub is not None) and val > ub:
                msg = f"Parameter '{param.name}' has value {val} above upper bound {ub}."
                raise ValueError(msg)
            
            if (lb is not None) and (ub is not None) and lb == ub:
                msg = f"Parameter '{param.name}' has identical lower and upper bounds {(lb, ub)}."
                raise ValueError(msg)


@dataclass(init=False, repr=False)
class SequentialModel:
    _model: Union[BaseModel, CompoundModel]

    n_submodels: int = field(init=False)
    submodels: tuple[BaseModel, ...] = field(init=False)

    _initial_values: NDArray[float64] = field(init=False)

    _params: tuple[Param, ...] = field(init=False)
    _n: int = field(init=False)
    _params_dict: dict[str, dict[str, int]] = field(init=False)

    # Free parameters
    _free_indices: NDArray[int_] = field(init=False)
    _free_params: tuple[Param, ...] = field(init=False)
    _n_free: int = field(init=False)

    # Tied parameters
    _tied_indices: NDArray[int_] = field(init=False)
    _tied_to_indices: NDArray[int_] = field(init=False)
    _tied_as: NDArray[float64] = field(init=False)
    _tied_bs: NDArray[float64] = field(init=False)
    _tied_params: tuple[Param, ...] = field(init=False)
    _n_tied: int = field(init=False)

    # Fixed parameters
    _fixed_indices: NDArray[int_] = field(init=False)
    _fixed_params: tuple[Param, ...] = field(init=False)
    _n_fixed: int = field(init=False)

    # Start/Stop
    _start_stop_indices: NDArray[int_] = field(init=False)

    def __init__(
        self,
        model: Union[BaseModel, CompoundModel],
    ) -> None:
        if isinstance(model, CompoundModel):
            _validate_compound_model(model)

        self._model = model
        self._initial_values = self._model.parameters.copy(order="C")
        self.n_submodels = self._model.n_submodels
        self.submodels = (
            (self._model,) if self.n_submodels == 1 else tuple(self._model)
        )

        # Parameters
        self._params = tuple(
            Param.from_parameter(getattr(submodel, param_name))
            for submodel in self.submodels
            for param_name in submodel.param_names
        )
        self._n = len(self._params)
        _validate_initial_values(self._params)

        self._prepare_params_dict()
        self._prepare_parameters()
        self._update_bounds()
        self._calculate_start_stop_indices()

    ## Useful dunder methods from Astropy's `CompoundModel`

    def __str__(self) -> str:
        return str(self._model)

    def __repr__(self) -> str:
        return repr(self._model)

    def __getitem__(self, idx: int | str) -> BaseModel:
        if self.n_submodels == 1:
            if idx == 0 or idx == self._model.name:
                return self._model
            else:
                raise IndexError(idx)
        else:
            return self._model[idx]

    @property
    def parameters(self) -> NDArray[float64]:
        return self._model.parameters

    @parameters.setter
    def parameters(self, values: NDArray[float64]) -> None:
        self._model.parameters = values
        for v, param in zip(values, self._params):
            param.value = v

    ## Free parameters

    def _prepare_params_dict(self) -> None:
        self._params_dict = {}
        count: int = 0
        for submodel in self.submodels:
            field: dict[str, int] = {}
            for param_name in submodel.param_names:
                field[param_name] = count
                count += 1

            self._params_dict[submodel.name] = field

    def _prepare_parameters(self) -> None:
        _free_indices: list[int] = []
        _free_params: list[Param] = []

        _tied_indices: list[int] = []
        _tied_to_indices: list[int] = []
        _tied_as: list[float] = []
        _tied_bs: list[float] = []
        _tied_params: list[Param] = []

        _fixed_indices: list[int] = []
        _fixed_params: list[Param] = []

        for i, param in enumerate(self._params):
            param_type = get_param_type(param, self.submodels, coerce=True)
            match param_type:
                case "free":
                    _free_indices.append(i)
                    _free_params.append(param)
                case "fixed":
                    _fixed_indices.append(i)
                    _fixed_params.append(param)
                case "tied":
                    _tied_indices.append(i)
                    _tied_params.append(param)

                    tied: LinearTie = param.tied
                    _tied_to_indices.append(
                        self._params_dict[tied.model_name][tied.parameter_name]
                    )
                    _tied_as.append(tied.a)
                    _tied_bs.append(tied.b)

        self._free_indices = array(_free_indices, dtype=int_)
        self._free_params = tuple(_free_params)
        self._n_free = len(self._free_params)

        self._tied_indices = array(_tied_indices, dtype=int_)
        self._tied_to_indices = array(_tied_to_indices, dtype=int_)
        self._tied_as = array(_tied_as, dtype=float64)
        self._tied_bs = array(_tied_bs, dtype=float64)
        self._tied_params = tuple(_tied_params)
        self._n_tied = len(self._tied_params)

        self._fixed_indices = array(_fixed_indices, dtype=int_)
        self._fixed_params = tuple(_fixed_params)
        self._n_fixed = len(self._fixed_params)

    def _calculate_start_stop_indices(self) -> None:
        self._start_stop_indices = empty(self.n_submodels + 1, dtype=int_)
        self._start_stop_indices[0] = 0
        for i, submodel in enumerate(self.submodels):
            n_params = len(submodel.param_names)
            self._start_stop_indices[i + 1] = self._start_stop_indices[i] + n_params

    def _update_bounds(self) -> None:
        for i, j in zip(self._tied_indices, self._tied_to_indices):
            limiter: Param = self._params[i]
            param: Param = self._params[j]
            
            a: float = limiter.tied.a
            b: float = limiter.tied.b
            if a == 0:
                continue

            _lb = limiter.bounds[0 if a > 0 else 1]
            _ub = limiter.bounds[1 if a > 0 else 0]

            new_lb, new_ub = param.bounds
            if isfinite(_lb):
                new_lb = max(new_lb, (_lb - b) / a)
            if isfinite(_ub):
                new_ub = min(new_ub, (_ub - b) / a)

            param.bounds = (new_lb, new_ub)
            param.value = apply_bounds.__wrapped__(param.value, param.bounds)
            self._initial_values[j] = param.value

    # Evaluation

    def _get_full_parameter_array(
        self, 
        free_params: NDArray[float64],
    ) -> NDArray[float64]:
        """
        Expands the truncated free parameter array to a full parameter array.

        This method uses the full array as the reference frame:
        1. Starts with initial parameter values
        2. Sets free parameters from the provided array
        3. Computes tied parameters using: tied_i = a_i * full[tied_to_i] + b_i

        Parameters
        ----------
        free_params : NDArray[float64]
            Array of free parameter values, length _n_free

        Returns
        -------
        NDArray[float64]
            Full parameter array of length _n
        """
        params = self._initial_values.copy()
        params[self._free_indices] = free_params
        if self._n_tied > 0:
            params[self._tied_indices] = (
                self._tied_as * params[self._tied_to_indices] 
                + self._tied_bs
            )
        return params

    def _map_free_params(self, params: NDArray[float64]) -> None:
        if len(params.shape) != 1:
            msg = f"Expected 1D array of free parameters, got {len(params.shape)}D array instead."
            raise ValueError(msg)
        if params.size != self._n_free:
            msg = f"Expected {self._n_free} free parameters, got {params.size} instead."
            raise ValueError(msg)

        for param, val in zip(self._free_params, params):
            param.value = val

    def evaluate(
        self, x: NDArray[float64], params: NDArray[float64]
    ) -> NDArray[float64]:
        """
        Evaluates the sequential model by summing contributions from all submodels.

        Uses in-place evaluation for efficiency: the output array `y` is passed to
        each submodel, which accumulates its contribution directly without creating
        intermediate arrays.

        Parameters
        ----------
        x : NDArray[float64]
            Wavelength array (1D)
        params : NDArray[float64]
            Free parameter values (length _n_free)

        Returns
        -------
        NDArray[float64]
            Combined model predictions, shape (len(x),)
        """
        y = zeros(x.size, dtype=float64)

        full_params = self._get_full_parameter_array(params)
        for i, submodel in enumerate(self.submodels):
            start, stop = self._start_stop_indices[i : i + 2]
            submodel.evaluate(x, *full_params[start:stop], y=y)

        return y

    def __call__(
        self,
        x: NDArray[float64],
    ) -> NDArray[float64]:
        return self.evaluate(
            x,
            self.parameters[self._free_indices],
        )

    def fit_deriv(
        self,
        x: NDArray[float64],
        params: NDArray[float64],
    ) -> list[NDArray[float64]]:
        return list(self.partial_deriv(x, params))

    def partial_deriv(
        self,
        x: NDArray[float64],
        params: NDArray[float64],
    ) -> NDArray[float64]:
        """
        Computes the Jacobian matrix of partial derivatives for all free parameters.

        For each free parameter, this includes contributions from submodels through
        tied parameters. When parameter y is tied to parameter x as y = a*x + b,
        the derivative is:

            ∂f_i/∂x = ∂M_i/∂x + a * ∂M_i/∂y

        This correction is applied after computing submodel derivatives using:
            full_derivs[tied_to] += a * full_derivs[tied]

        Parameters
        ----------
        x : NDArray[float64]
            Wavelength array (1D)
        params : NDArray[float64]
            Free parameter values (length _n_free)

        Returns
        -------
        NDArray[float64]
            Jacobian matrix, shape (_n_free, len(x))
        """
        full_derivs = zeros((self._n, x.size), dtype=float64)
        full_params = self._get_full_parameter_array(params)

        for submodel, start, stop in zip(
            self.submodels,
            self._start_stop_indices[:-1],
            self._start_stop_indices[1:],
        ):
            submodel.partial_deriv(
                x,
                *full_params[start:stop],
                derivs=full_derivs[start:stop],
            )

        full_derivs[self._tied_to_indices] += (
            self._tied_as[:, None] * full_derivs[self._tied_indices]
        )

        return full_derivs[self._free_indices]

    def get_final_model(
        self,
        params: NDArray[float64] | None = None,
        copy: bool = True,
    ) -> Union[BaseModel, CompoundModel]:
        """
        Returns an Astropy (compound) model instance with the final fitted 
        parameter values.

        Parameters
        ----------
        params : NDArray[float64] | None
            Free parameter values (length _n_free). If None, uses current values.
        copy : bool
            If True, returns a copy of the Astropy model used to instantiate
            this SequentialModel. Otherwise, returns the original, modified
            Astropy model.

        Returns
        -------
        BaseModel | CompoundModel
            Model instance with updated parameter values.
        """
        out = self._model.copy() if copy else self._model
        if params is None:
            return out
        
        if params.size != self._n:
            params = self._get_full_parameter_array(params)

        out.parameters = params
        if not copy:
            # Update own parameters
            self.parameters = params

        return out

    def get_updated_model(
        self,
        copy: bool = True,
    ) -> Union[BaseModel, CompoundModel]:
        return self.get_final_model(
            self.parameters[self._free_indices],
            copy=copy,
        )

    ### For Scipy `least_squares` optimization

    def fun(
        self, 
        x: NDArray[float64], 
        data: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        z = self.evaluate(data["x"], x)
        z -= data["y"]
        return einsum("i,i->i", z, data["w"], out=z)

    def jac(
        self, 
        x: NDArray[float64], 
        data: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        jac = self.partial_deriv(data["x"], x).T
        return einsum("ij,i->ij", jac, data["w"], out=jac)

    @property
    def x0(self) -> NDArray[float64]:
        return self._initial_values[self._free_indices]

    @property
    def bounds(self) -> tuple[NDArray[float64], NDArray[float64]]:
        return tuple(
            fromiter(
                (param.bounds[i] for param in self._free_params), 
                dtype=float64,
            ) 
            for i in (0, 1)
        )
    