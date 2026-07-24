from typing import Iterable, Union
from warnings import warn
from functools import cached_property
from collections import Counter
from numpy.typing import NDArray
from numpy import float64, isfinite, zeros, empty, int_, arange, fromiter, inf, einsum

from astropy.modeling import Parameter, CompoundModel

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

        if m.op != '+':
            msg = "Compound model must be constructed using only the '+' operator." 
            raise ValueError(msg)
        
        m = m.left

def _validate_initial_values(params: Iterable[Parameter]) -> None:
    for param in params:
        if not isfinite(val := param.value):
            msg = "Parameter '{}' has non-finite value: {}."\
                .format(param.name, val)
            raise ValueError(msg)
        
        if not param.fixed:
            lb, ub = param.bounds
            if (lb is not None) and val < lb:
                msg = "Parameter '{}' has value {} below lower bound {}."\
                    .format(param.name, val, lb)
                raise ValueError(msg)
            if (ub is not None) and val > ub:
                msg = "Parameter '{}' has value {} above upper bound {}."\
                    .format(param.name, val, ub)
                raise ValueError(msg)
            if (lb is not None) and (ub is not None) and lb == ub:
                msg = "Parameter '{}' has identical lower and upper bounds {}."\
                    .format(param.name, (lb, ub))
                raise ValueError(msg)

def _validate_tied_parameters(params: Iterable[Parameter]) -> None:
    for param in params:
        if (tied := param.tied) in (False, None):
            continue
        elif not isinstance(tied, LinearTie):
            msg = "Parameter '{}' has invalid 'tied' attribute: {}."\
                .format(param.name, tied)
            raise TypeError(msg)
        elif param.fixed:
            msg = "Parameter '{}' is fixed but has a 'tied' attribute: {}."\
                .format(param.name, tied)
            warn(msg, UserWarning, stacklevel=2)

def _validate_unique_submodel_names(submodels: Iterable[BaseModel]) -> None:
    name_counts = Counter(submodel.name for submodel in submodels)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        msg = "Submodels must have unique names. Found duplicates: {}."\
            .format(duplicates)
        raise ValueError(msg)

class SequentialModel:
    def __init__(
        self,
        model: Union[BaseModel, CompoundModel],
    ) -> None:
        if isinstance(model, CompoundModel):
            _validate_compound_model(model)

        self._model: Union[BaseModel, CompoundModel] = model
        self._initial_values: NDArray[float64] = self._model.parameters.copy()
        self._validate_params()

        # Preparation
        self._calculate_start_stop_indices()
        self._calculate_free_indices()
        self._calculate_tied_indices()

    def _validate_submodels(self) -> None:
        _validate_unique_submodel_names(self.submodels)

    def _validate_params(self) -> None:
        """
        Validates the following:
        1.  All parameters' initial values are finite, and lie within any given 
            bounds. These bounds should not be identical.  
        2.  All parameters' 'tied' attributes are either False, None, or an 
            instance of 'LinearTie'. If a 'tied' parameter is fixed, a warning 
            is raised. 
        """
        _validate_initial_values(self._parameters)
        _validate_tied_parameters(self._parameters)
    
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
        
    @cached_property
    def n_submodels(self) -> int:
        return self._model.n_submodels
    
    @cached_property
    def submodels(self) -> tuple[BaseModel]:
        if self.n_submodels == 1:
            return (self._model,)
        else:
            return tuple(self._model.submodels)

    # Parameters

    @property
    def parameters(self) -> NDArray[float64]:
        return self._model.parameters

    @cached_property
    def _parameters(self) -> tuple[Parameter]:
        return tuple(
            getattr(submodel, param_name)
            for submodel in self.submodels
            for param_name in submodel.param_names
        )

    @property
    def _n(self) -> int:
        return len(self._parameters)

    @cached_property
    def _parameters_dict(self) -> dict[str, dict[str, tuple[int, Parameter]]]:
        """
        Returns a dictionary of dictionaries mapping from submodel and parameter 
        names to a tuple of the parameter instance and the parameter's 
        corresponding index in the full parameter array.
        """
        _parameters_dict = {}
        count: int = 0
        for submodel in self.submodels:
            field = {}
            for param_name in submodel.param_names:
                param = getattr(submodel, param_name)
                field[param_name] = (count, param)
                count += 1

            _parameters_dict[submodel.name] = field

        return _parameters_dict

    @cached_property
    def _free_parameters_dict(self) -> dict[str, dict[str, tuple[int, Parameter]]]:
        """
        Returns a dictionary of dictionaries mapping from submodel and parameter 
        names to a tuple of the parameter instance and the parameter's 
        corresponding index in the truncated free parameter array.
        """
        _parameters_dict = {}
        count: int = 0
        for submodel in self.submodels:
            field = {}
            for param_name in submodel.param_names:
                param = getattr(submodel, param_name)
                if not self._param_is_free(param):
                    continue

                field[param_name] = (count, param)
                count += 1

            _parameters_dict[submodel.name] = field

        return _parameters_dict

    ## Free parameters

    def _param_is_free(self, param: Parameter) -> bool:
        """
        Returns True if the parameter is not fixed and not tied to another 
        non-fixed parameter.
        """
        return not param.fixed and not self._param_is_tied(param)

    @cached_property
    def _free_parameters(self) -> tuple[Parameter, ...]:
        return tuple(filter(self._param_is_free, self._parameters))

    @property
    def _n_free(self) -> int:
        return len(self._free_parameters)

    ## Tied parameters

    def _param_is_tied(self, param: Parameter) -> bool:
        """
        Returns True if the parameter is not fixed and is tied to another 
        non-fixed parameter.
        """
        if param.fixed or not param.tied:
            return False

        tied: LinearTie = param.tied
        for submodel in self.submodels:
            if submodel.name == tied.model_name:
                tied_param: Parameter = getattr(submodel, tied.parameter_name)
                if tied_param.tied not in (False, None):
                    msg = "Parameter '{}' is tied to another tied parameter '{}'. "\
                          "Tie chains are not allowed.".format(param.name, tied_param.name)
                    raise ValueError(msg)
                return not tied_param.fixed

        raise ValueError("Could not find submodel: {}".format(tied.model_name))

    @cached_property
    def _tied_parameters(self) -> tuple[Parameter, ...]:
        return tuple(filter(self._param_is_tied, self._parameters))

    @property
    def _n_tied(self) -> int:
        return len(self._tied_parameters)

    ## Fixed parameters

    def _param_is_fixed(self, param: Parameter) -> bool:
        """
        Returns True if the parameter is fixed, or if it is tied to a fixed
        parameter.
        """
        return param.fixed or (param.tied and not self._param_is_tied(param))

    @cached_property
    def _fixed_parameters(self) -> tuple[Parameter, ...]:
        return tuple(filter(self._param_is_fixed, self._parameters))

    @property
    def _n_fixed(self) -> int:
        return len(self._fixed_parameters)

    # Preparation

    def _calculate_free_indices(self) -> None:
        """
        Calculates a vector of indices that maps the truncated free parameter 
        array onto the full parameter array. In the event that all parameters 
        are free, the map is simply an array of integers from '0' to 'n-1', 
        where 'n' is the total number of parameters.
        """
        if self._n == self._n_free:
            self._free_indices = arange(self._n, dtype=int_)
        else:
            self._free_indices = fromiter(
                map(self._parameters.index, self._free_parameters),
                dtype=int_,
            )

    def _calculate_tied_indices(self) -> None:
        """
        Calculates the following vectors:
        1.  A vector of indices that extracts the tied value from the truncated free parameter array.
        2.  A vector of indices that maps a value onto the full parameter array.
        3.  A vector of 'a' coefficients for the tied parameters.
        4.  A vector of 'b' coefficients for the tied parameters.
        """
        self._tied_indices = empty(self._n_tied, dtype=int_)
        self._tied_to_indices = empty(self._n_tied, dtype=int_)
        self._tied_as = empty(self._n_tied, dtype=float64)
        self._tied_bs = empty(self._n_tied, dtype=float64)

        if self._n_tied > 0:
            i: int = 0
            for j, param in enumerate(self._parameters):
                if param not in self._tied_parameters:
                    continue

                self._tied_indices[i] = j

                tied: LinearTie = param.tied
                self._tied_to_indices[i] = self._parameters_dict[tied.model_name][tied.parameter_name][0]
                self._tied_as[i] = tied.a
                self._tied_bs[i] = tied.b
                i += 1
                
    def _calculate_start_stop_indices(self) -> None:
        self._start_stop_indices = empty(self.n_submodels + 1, dtype=int_)
        self._start_stop_indices[0] = 0
        for i, submodel in enumerate(self.submodels):
            n_params = len(submodel.param_names)
            self._start_stop_indices[i+1] = self._start_stop_indices[i] + n_params

    @cached_property
    def _not_fixed_submodels(self) -> list[BaseModel]:
        """
        Creates a list of submodel that aren't fully fixed.
        """
        def func(submodel: BaseModel) -> bool:
            return any(
                not getattr(submodel, param_name).fixed
                for param_name in submodel.param_names
            )
        return list(filter(func, self.submodels))

    # Evaluation

    def _get_full_parameter_array(self, free_params: NDArray[float64]) -> NDArray[float64]:
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
        if not len(params.shape) == 1:
            msg = "Expected 1D array of free parameters, got {}D array instead."\
                .format(len(params.shape))
            raise ValueError(msg)
        if params.size != self._n_free:
            msg = "Expected {} free parameters, got {} instead."\
                .format(self._n_free, params.size)
            raise ValueError(msg)
        
        for param, val in zip(self._free_parameters, params):
            param.value = val

    def evaluate(self, x: NDArray[float64], params: NDArray[float64]) -> NDArray[float64]:
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
            start, stop = self._start_stop_indices[i:i+2]
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

        full_derivs[self._tied_to_indices] += self._tied_as * full_derivs[self._tied_indices]

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
                if params.size != self._n else
                params
            )
        return out

    ### For Scipy `least_squares` optimization

    def fun(self, x: NDArray[float64], data: dict[str, NDArray[float64]]) -> NDArray[float64]:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        z = self.evaluate(data['x'], x)
        z -= data['y']
        return einsum('i,i->i', z, data['w'], out=z)

    def jac(self, x: NDArray[float64], data: dict[str, NDArray[float64]]) -> NDArray[float64]:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        jac = self.partial_deriv(data['x'], x)
        return einsum('ij,i->ij', jac, data['w'], out=jac)

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