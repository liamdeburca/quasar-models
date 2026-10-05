from dataclasses import field
from typing import Union

from numpy import array_equal, bool_, einsum, float64, ones, stack
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import CompoundModel_
from quasar_typing.numpy import BoolVector, FloatMatrix, FloatVector, IntVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.interpolation import create_interp_matrix
from scipy.sparse import eye

from .base_model import BaseModel
from .sequential_model import SequentialModel
from .utils import Param


@dataclass(repr=False)
class InterpolatedSequentialModel:
    sequential_model: SequentialModel

    x: FloatVector = field(kw_only=True)    
    x_new: FloatVector = field(kw_only=True)

    _interp_matrix: csr_matrix_ = field(init=False)
    _mask: BoolVector = field(init=False)

    def __post_init__(self):
        """
        Sets the following attributes based on the input `x` and `x_new` arrays:

        - `_interp_matrix`: sparse matrix for interpolating values from `x` to 
            `x_new`. If `x` and `x_new` are identical, the interpolation matrix 
            is the identity matrix.
        - `_mask`: boolean array indicating which elements of `x_new` fall 
            within the range of `x`. This ensures that data-model comparisons 
            are only performed where the model can be evaluated. If `x` and 
            `x_new` are identical, the mask is all `True`.
        """
        if array_equal(self.x, self.x_new):
            n = self.x.size
            self._interp_matrix = eye(n, n, dtype=float64, format="csr")
            self._mask = ones(n, dtype=bool_)
        else:
            self._interp_matrix = create_interp_matrix.__wrapped__(
                self.x, self.x_new,
            )[0]
            lb = self.x.min()
            ub = self.x.max()
            self._mask = (lb <= self.x_new) & (self.x_new <= ub)

            if not self._mask.any():
                raise ValueError(
                    "No elements of `x_new` fall within the range of `x`."
                )

    @property
    def n_submodels(self) -> int:
        return self.sequential_model.n_submodels

    @property
    def submodels(self) -> tuple[BaseModel, ...]:
        return self.sequential_model.submodels

    @property
    def _initial_values(self) -> FloatVector:
        return self.sequential_model._initial_values

    @property
    def _params(self) -> tuple[Param, ...]:
        return self.sequential_model._params

    @property
    def _n(self) -> int:
        return self.sequential_model._n

    @property
    def _params_dict(self) -> dict[str, dict[str, int]]:
        return self.sequential_model._params_dict

    @property
    def _free_indices(self) -> IntVector:
        return self.sequential_model._free_indices

    @property
    def _free_params(self) -> tuple[Param, ...]:
        return self.sequential_model._free_params

    @property
    def _n_free(self) -> int:
        return self.sequential_model._n_free

    @property
    def _tied_indices(self) -> IntVector:
        return self.sequential_model._tied_indices

    @property
    def _tied_to_indices(self) -> IntVector:
        return self.sequential_model._tied_to_indices

    @property
    def _tied_as(self) -> FloatVector:
        return self.sequential_model._tied_as

    @property
    def _tied_bs(self) -> FloatVector:
        return self.sequential_model._tied_bs

    @property
    def _tied_params(self) -> tuple[Param, ...]:
        return self.sequential_model._tied_params

    @property
    def _n_tied(self) -> int:
        return self.sequential_model._n_tied

    @property
    def _fixed_indices(self) -> IntVector:
        return self.sequential_model._fixed_indices

    @property
    def _fixed_params(self) -> tuple[Param, ...]:
        return self.sequential_model._fixed_params

    @property
    def _n_fixed(self) -> int:
        return self.sequential_model._n_fixed

    @property
    def _start_stop_indices(self) -> IntVector:
        return self.sequential_model._start_stop_indices

    @property
    def parameters(self) -> FloatVector:
        return self.sequential_model.parameters

    def evaluate(self, params: FloatVector) -> FloatVector:
        y = self.sequential_model.evaluate(self.x, params)
        return y @ self._interp_matrix

    def __call__(self) -> FloatVector:
        return self.evaluate(self.parameters[self._free_indices])

    def fit_deriv(self, params: FloatVector) -> list[FloatVector]:
        return list(self.partial_deriv(self.x, params))

    def partial_deriv(self, params: FloatVector) -> FloatMatrix:
        derivs = self.sequential_model.partial_deriv(self.x, params)
        return stack(
            [d @ self._interp_matrix for d in derivs],
            axis=0,
        )

    def get_final_model(
        self,
        params: FloatVector | None = None,
        copy: bool = True,
    ) -> Union[BaseModel, CompoundModel_[BaseModel]]:
        return self.sequential_model.get_final_model(params=params, copy=copy)

    def get_updated_model(
        self,
        copy: bool = True,
    ) -> Union[BaseModel, CompoundModel_[BaseModel]]:
        return self.sequential_model.get_updated_model(copy=copy)

    ### For Scipy `least_squares` optimization

    def fun(
        self, 
        x: FloatVector,
        data: dict[str, FloatVector],
    ) -> FloatVector:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        z = self.evaluate(x)[self._mask]
        z -= data["y"][self._mask]
        return einsum("i,i->i", z, data["w"][self._mask], out=z)

    def jac(
        self, 
        x: FloatVector, 
        data: dict[str, FloatVector],
    ) -> FloatMatrix:
        """
        Scipy notation: 'x' is the vector of free parameters.
        """
        jac = self.partial_deriv(x)[:,self._mask].T
        return einsum("ij,i->ij", jac, data["w"][self._mask], out=jac)

    @property
    def x0(self) -> FloatVector:
        return self.sequential_model.x0

    @property
    def bounds(self) -> tuple[FloatVector, FloatVector]:
        return self.sequential_model.bounds    