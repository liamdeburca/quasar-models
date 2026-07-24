from typing import Literal
from dataclasses import field
from pydantic.dataclasses import dataclass

@dataclass(kw_only=True)
class LinearTie:
    a: float
    b: float

    model_name: str
    parameter_name: str

    idx: int | None = field(default=None, init=False)

    def __bool__(self) -> Literal[True]:
        return True