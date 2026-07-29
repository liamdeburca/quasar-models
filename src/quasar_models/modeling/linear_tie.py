from dataclasses import field
from typing import Literal

from astropy.modeling import CompoundModel
from pydantic.dataclasses import dataclass


@dataclass(kw_only=True)
class LinearTie:
    a: float
    b: float

    model_name: str
    parameter_name: str

    def __bool__(self) -> Literal[True]:
        return True

    def __call__(self, model: CompoundModel) -> float:
        x = getattr(model[self.model_name], self.parameter_name).value
        return self.a * x + self.b