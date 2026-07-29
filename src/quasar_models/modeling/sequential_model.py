from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union
from warnings import warn

from astropy.modeling import CompoundModel, Parameter
from numpy import (
    array,
    einsum,
    empty,
    float64,
    inf,
    int_,
    isfinite,
    zeros,
)
from numpy.typing import NDArray

from .base_model import BaseModel
from .linear_tie import LinearTie


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


def _validate_initial_values(params: Iterable[Parameter]) -> None:
    for param in params:
        if not isfinite(val := param.value):
            msg = f"Parameter '{param.name}' has non-finite value: {val}."
            raise ValueError(msg)

        if not param.fixed:
            lb, ub = param.bounds
            if (lb is not None) and val < lb:
                msg = (
                    f"Parameter '{param.name}' has value {val} below lower bound {lb}."
                )
                raise ValueError(msg)
            if (ub is not None) and val > ub:
                msg = (
                    f"Parameter '{param.name}' has value {val} above upper bound {ub}."
                )
                raise ValueError(msg)
            if (lb is not None) and (ub is not None) and lb == ub:
                msg = f"Parameter '{param.name}' has identical lower and upper bounds {(lb, ub)}."
                raise ValueError(msg)


def _validate_tied_parameters(params: Iterable[Parameter]) -> None:
    for param in params:
        if (tied := param.tied) in (False, None):
            continue
        elif not isinstance(tied, LinearTie):
            msg = f"Parameter '{param.name}' has invalid 'tied' attribute: {tied}."
            raise TypeError(msg)
        elif param.fixed:
            msg = (
                f"Parameter '{param.name}' is fixed but has a 'tied' attribute: {tied}."
            )
            warn(msg, UserWarning, stacklevel=2)


def _validate_unique_submodel_names(submodels: Iterable[BaseModel]) -> None:
    name_counts = Counter(submodel.name for submodel in submodels)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        msg = f"Submodels must have unique names. Found duplicates: {duplicates}."
        raise ValueError(msg)


@dataclass(init=False, repr=False)
class SequentialModel:
    _model: Union[BaseModel, CompoundModel]

    n_submodels: int = field(init=False)
    submodels: tuple[BaseModel, ...] = field(init=False)

    _initial_values: NDArray[float64] = field(init=False)

    _parameters: tuple[Parameter, ...] = field(init=False)
    _n: int = field(init=False)
    _parameters_dict: dict[str, dict[str, tuple[int, Parameter]]] = field(init=False)

    # Free parameters
    _free_indices: NDArray[int_] = field(init=False)
    _free_parameters: tuple[Parameter, ...] = field(init=False)
    _n_free: int = field(init=False)

    # Tied parameters
    _tied_indices: NDArray[int_] = field(init=False)
    _tied_to_indices: NDArray[int_] = field(init=False)
    _tied_as: NDArray[float64] = field(init=False)
    _tied_bs: NDArray[float64] = field(init=False)
    _tied_parameters: tuple[Parameter, ...] = field(init=False)
    _n_tied: int = field(init=False)

    # Fixed parameters
    _fixed_indices: NDArray[int_] = field(init=False)
    _fixed_parameters: tuple[Parameter, ...] = field(init=False)
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
        self._initial_values = self._model.parameters.copy()
        self.n_submodels = self._model.n_submodels
        self.submodels = (
            (self._model,) if self.n_submodels == 1 else tuple(self._model)
        )
        self._validate_submodels(self.submodels)

        # Parameters

        self._parameters = tuple(
            getattr(submodel, param_name)
            for submodel in self.submodels
            for param_name in submodel.param_names
        )
        self._validate_params(self._parameters)
        self._n = len(self._parameters)

        count: int = 0
        self._parameters_dict = {}
        for submodel in self.submodels:
            field = {}
            for param_name in submodel.param_names:
                param = getattr(submodel, param_name)

                field[param_name] = (count, param)
                count += 1

            self._parameters_dict[submodel.name] = field

        # Preparation
        self._prepare_free()
        self._prepare_tied()
        self._prepare_fixed()
        self._calculate_start_stop_indices()

    @staticmethod
    def _validate_submodels(submodels: tuple[BaseModel, ...]) -> None:
        _validate_unique_submodel_names(submodels)

    @staticmethod
    def _validate_params(parameters: tuple[Parameter, ...]) -> None:
        """
        Validates the following:
        1.  All parameters' initial values are finite, and lie within any given
            bounds. These bounds should not be identical.
        2.  All parameters' 'tied' attributes are either False, None, or an
            instance of 'LinearTie'. If a 'tied' parameter is fixed, a warning
            is raised.
        """
        _validate_initial_values(parameters)
        _validate_tied_parameters(parameters)

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

    ## Free parameters

    def _param_is_free(self, param: Parameter) -> bool:
        """
        Returns True if the parameter is not fixed and not tied to another
        non-fixed parameter.
        """
        return not param.fixed and not self._param_is_tied(param)

    def _prepare_free(self) -> None:
        _free_indices: list[int] = []
        _free_parameters: list[Parameter] = []
        for i, param in filter(
            lambda tup: self._param_is_free(tup[1]),
            enumerate(self._parameters),
        ):
            _free_indices.append(i)
            _free_parameters.append(param)

        self._free_indices = array(_free_indices, dtype=int_)
        self._free_parameters = tuple(_free_parameters)
        self._n_free = len(self._free_parameters)

    ## Tied parameters

    def _param_is_tied(self, param: Parameter) -> bool:
        """
        Returns True if the parameter is not fixed and is tied to another
        non-fixed parameter.
        """
        if not isinstance(tied := param.tied, LinearTie):
            return False

        for submodel in self.submodels:
            if submodel.name == tied.model_name:
                tied_param: Parameter = getattr(submodel, tied.parameter_name)
                if tied_param.tied not in (False, None):
                    msg = (
                        f"Parameter '{param.name}' is tied to another tied parameter '{tied_param.name}'. "
                        "Tie chains are not allowed."
                    )
                    raise ValueError(msg)
                return not tied_param.fixed

        raise ValueError(f"Could not find submodel: {tied.model_name}")

    def _prepare_tied(self) -> None:
        _tied_indices: list[int] = []
        _tied_to_indices: list[int] = []
        _tied_as: list[float] = []
        _tied_bs: list[float] = []
        _tied_parameters: list[Parameter] = []

        for i, param in filter(
            lambda tup: self._param_is_tied(tup[1]),
            enumerate(self._parameters),
        ):
            _tied_indices.append(i)
            _tied_parameters.append(param)
            
            tied: LinearTie = param.tied
            _tied_to_indices.append(
                self._parameters_dict[tied.model_name][tied.parameter_name][0]
            )
            _tied_as.append(tied.a)
            _tied_bs.append(tied.b)

        self._tied_indices = array(_tied_indices, dtype=int_)
        self._tied_to_indices = array(_tied_to_indices, dtype=int_)
        self._tied_as = array(_tied_as, dtype=float64)
        self._tied_bs = array(_tied_bs, dtype=float64)
        self._tied_parameters = tuple(_tied_parameters)
        self._n_tied = len(self._tied_parameters)

    ## Fixed parameters

    def _param_is_fixed(self, param: Parameter) -> bool:
        """
        Returns True if the parameter is fixed, or if it is tied to a fixed
        parameter.
        """
        return param.fixed or (param.tied and not self._param_is_tied(param))

    def _prepare_fixed(self) -> None:
        _fixed_indices: list[int] = []
        _fixed_parameters: list[Parameter] = []
        for i, param in filter(
            lambda tup: self._param_is_fixed(tup[1]),
            enumerate(self._parameters),
        ):
            _fixed_indices.append(i)
            _fixed_parameters.append(param)

        self._fixed_indices = array(_fixed_indices, dtype=int_)
        self._fixed_parameters = tuple(_fixed_parameters)
        self._n_fixed = len(self._fixed_parameters)

    # Preparation

    def _calculate_start_stop_indices(self) -> None:
        self._start_stop_indices = empty(self.n_submodels + 1, dtype=int_)
        self._start_stop_indices[0] = 0
        for i, submodel in enumerate(self.submodels):
            n_params = len(submodel.param_names)
            self._start_stop_indices[i + 1] = self._start_stop_indices[i] + n_params

    # Evaluation

    def _get_full_parameter_array(
        self, free_params: NDArray[float64]
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
                self._tied_as * params[self._tied_to_indices] + self._tied_bs
            )
        return params

    def _map_free_params(self, params: NDArray[float64]) -> None:
        if len(params.shape) != 1:
            msg = f"Expected 1D array of free parameters, got {len(params.shape)}D array instead."
            raise ValueError(msg)
        if params.size != self._n_free:
            msg = f"Expected {self._n_free} free parameters, got {params.size} instead."
            raise ValueError(msg)

        for param, val in zip(self._free_parameters, params):
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
        Returns a new model instance with the final fitted parameter values.

        Parameters
        ----------
        params : NDArray[float64] | None
            Free parameter values (length _n_free). If None, uses current values.
        copy : bool
            If True, returns a copy of the model; otherwise, returns the original.

        Returns
        -------
        BaseModel | CompoundModel
            Model instance with updated parameter values.
        """
        out = self._model.copy() if copy else self._model
        if params is not None:
            out.parameters = (
                self._get_full_parameter_array(params)
                if params.size != self._n
                else params
            )
        return out

    ### For Scipy `least_squares` optimization

    def fun(
        self, x: NDArray[float64], data: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        z = self.evaluate(data["x"], x)
        z -= data["y"]
        return einsum("i,i->i", z, data["w"], out=z)

    def jac(
        self, x: NDArray[float64], data: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        jac = self.partial_deriv(data["x"], x)
        return einsum("ij,i->ij", jac, data["w"], out=jac)

    @property
    def x0(self) -> NDArray[float64]:
        return self._initial_values[self._free_indices]

    @property
    def bounds(self) -> tuple[NDArray[float64], NDArray[float64]]:
        lbs = empty(self._n_free, dtype=float64)
        ubs = empty(self._n_free, dtype=float64)
        for i, param in enumerate(self._free_parameters):
            lbs[i] = -inf if param.bounds[0] is None else param.bounds[0]
            ubs[i] = inf if param.bounds[1] is None else param.bounds[1]

        return (lbs, ubs)
