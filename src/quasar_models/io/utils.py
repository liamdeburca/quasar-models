from collections.abc import Callable
from math import inf
from typing import TypedDict

from astropy.modeling import Parameter
from astropy.units import Unit

from quasar_models.modeling.utils import LinearTie


class SerializedTied(TypedDict):
    a: float
    b: float
    model_name: str
    parameter_name: str


def serialize_tied(tied: Callable | LinearTie | None) -> SerializedTied | None:
    if (tied is None) or not isinstance(tied, LinearTie):
        return None
    return {
        "a": tied.a, 
        "b": tied.b,
        "model_name": tied.model_name, 
        "parameter_name": tied.parameter_name,
    }


def deserialize_tied(serialized: SerializedTied | None) -> LinearTie | None:
    if serialized is None:
        return None
    return LinearTie(
        a=serialized["a"],
        b=serialized["b"],
        model_name=serialized["model_name"],
        parameter_name=serialized["parameter_name"],
    )

###

class SerializedParameter(TypedDict):
    value: float
    unit: str | None
    fixed: bool
    bounds: tuple[float, float]
    tied: SerializedTied | None

def serialize_parameter(
    parameter: Parameter, 
    unit: str = "",
) -> SerializedParameter:
    lb = -inf if parameter.bounds[0] is None else parameter.bounds[0]
    ub = inf if parameter.bounds[1] is None else parameter.bounds[1]
    return {
        "value": parameter.value,
        "unit": unit or None,
        "fixed": parameter.fixed,
        "bounds": (lb, ub),
        "tied": serialize_tied(parameter.tied),
    }

def deserialize_parameter(param: Parameter, serialized: SerializedParameter, unit: str | None = "") -> None:
    if (not unit) and (not serialized["unit"]):
        # No units
        def converter(value: float | None) -> float | None:
            return value
    elif unit and serialized["unit"]:
        # Yes units, for both
        def converter(value: float | None) -> float | None:
            if value is None:
                return None
            return (value * Unit(serialized["unit"])).to(unit).value
    else:
        raise ValueError(f"Unit mismatch: expected '{unit}', serialized '{serialized['unit']}'")

    param.value = converter(serialized["value"])
    param.fixed = bool(serialized["fixed"])
    param.bounds = tuple(map(converter, serialized["bounds"]))
    param.tied = deserialize_tied(serialized["tied"])