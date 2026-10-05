
from pydantic.dataclasses import dataclass
from quasar_typing.numpy import BoolVector, FloatMatrix, FloatVector

from .sequential_model import SequentialModel


@dataclass(kw_only=True)
class PreparedSequentialModel:
    """
    Wrapper class for `SequentialModel`.
    
    Includes the data to fit, including the wavelength grid used to prepare the 
    model, and a `where` mask indicating which data points to fit to.
    """
    seq_model: SequentialModel
    data: dict[str, FloatVector]
    where: BoolVector | None

    @property
    def x(self) -> FloatVector:
        return self.data['x']

    ### w/o instrumental correction

    def _evaluate(self, params: FloatVector) -> FloatVector:
        return self.seq_model._evaluate(self.x, params)

    def _partial_deriv(self, params: FloatVector) -> FloatMatrix:
        return self.seq_model._partial_deriv(self.x, params)

    def _fit_deriv(self, params: FloatVector) -> list[FloatVector]:
        return self.seq_model._fit_deriv(self.x, params)

    ### w/ instrumental correction

    def evaluate(self, params: FloatVector) -> FloatVector:
        return self.seq_model.evaluate(self.x, params)

    def partial_deriv(self, params: FloatVector) -> FloatMatrix:
        return self.seq_model.partial_deriv(self.x, params)

    def fit_deriv(self, params: FloatVector) -> list[FloatVector]:
        return self.seq_model.fit_deriv(self.x, params)

    ### For Scipy `least_squares` optimization

    def fun(self, x: FloatVector) -> FloatVector:
        return self.seq_model.fun(x, self.data, where=self.where)

    def jac(self, x: FloatVector) -> FloatMatrix:
        return self.seq_model.jac(x, self.data, where=self.where)

    @property
    def x0(self) -> FloatVector:
        return self.seq_model.x0

    @property
    def bounds(self) -> tuple[FloatVector, FloatVector]:
        return self.seq_model.bounds
