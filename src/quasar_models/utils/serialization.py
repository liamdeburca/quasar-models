"""
Utilities for serializing and deserializing custom Astropy-compatible models.
"""
from collections.abc import Callable
from typing import Any

from astropy.modeling import Parameter
from astropy.units import Unit
from numpy import arange, array, diff, isclose
from quasar_typing.numpy import Array_

from ..modeling.utils import LinearTie


def serialize_quantity(value: float, unit: str) -> dict[str, Any]:
    return {
        "value": float(value),
        "unit": str(unit),
    }

def deserialize_quantity(
    data: dict[str, Any], 
    unit: str | None,
) -> float:
    value = float(data["value"])
    if unit is None:
        return value
    return (value * Unit(data["unit"])).to(unit).value

###

def serialize_array(value: Array_, unit: str | None) -> dict[str, Any]:
    if value.size > 5:
        delta = diff(value)
        if isclose(delta.std(), 0.0):
            return serialize_lin_array(value, unit)
        if isclose((delta / value[:-1]).std(), 0.0):
            return serialize_log_array(value, unit)
    return serialize_any_array(value, unit)

def deserialize_array(data: dict[str, Any], unit: str | None) -> Array_:
    if data.get("lin"):
        return deserialize_lin_array(data, unit)
    if data.get("log"):
        return deserialize_log_array(data, unit)
    return deserialize_any_array(data, unit)

###

def serialize_any_array(value: Array_, unit: str | None) -> dict[str, Any]:
    out = {
        "value": value.tolist(),
        "dtype": str(value.dtype),
        "readonly": not value.flags["WRITEABLE"]
    }
    if unit is not None:
        out["unit"] = unit
    return out

def deserialize_any_array(data: dict[str, Any], unit: str | None) -> Array_:
    value = array(data["value"], dtype=data["dtype"], order="C")
    value.setflags(write=not data["readonly"])
    if unit is not None:
        value = (value * Unit(data["unit"])).to(unit).value
    return value


def serialize_log_array(value: Array_, unit: str | None) -> dict[str, Any]:
    out = {
        "log": True,
        "value": value[0],
        "dv": value[1] / value[0] - 1.0,
        "n": value.size,
        "dtype": str(value.dtype),
        "readonly": not value.flags["WRITEABLE"]
    }
    if unit is not None:
        out["unit"] = unit
    return out

def deserialize_log_array(data: dict[str, Any], unit: str | None) -> Array_:
    assert data.get("log")
    value = array(
        data["value"] * (1 + data["dv"]) ** arange(data["n"]),
        dtype=data["dtype"],
        order="C"
    )
    value.setflags(write=not data["readonly"])
    if unit is not None:
        value = (value * Unit(data["unit"])).to(unit).value
    return value


def serialize_lin_array(value: Array_, unit: str | None) -> dict[str, Any]:
    out = {
        "lin": True,
        "value": value[0],
        "dx": value[1] - value[0],
        "n": value.size,
        "dtype": str(value.dtype),
        "readonly": not value.flags["WRITEABLE"]
    }
    if unit is not None:
        out["unit"] = unit
    return out

def deserialize_lin_array(data: dict[str, Any], unit: str | None) -> Array_:
    assert data.get("lin")
    value = array(
        data["value"] + data["dx"] * arange(data["n"]),
        dtype=data["dtype"],
        order="C"
    )
    value.setflags(write=not data["readonly"])
    if unit is not None:
        value = (value * Unit(data["unit"])).to(unit).value
    return value

###

def serialize_tied(tied: Callable | LinearTie | None) -> dict[str, Any]:
    if (tied is None) or not isinstance(tied, LinearTie):
        return {"tied": None}
    return {
        "tied": {
            "a": float(tied.a),
            "b": float(tied.b),
            "model_name": str(tied.model_name), 
            "parameter_name": str(tied.parameter_name),
        }
    }


def deserialize_tied(data: dict[str, Any]) -> LinearTie | None:
    tied_data = data.get("tied")
    if tied_data is None:
        return None
    return LinearTie(
        a=float(tied_data["a"]),
        b=float(tied_data["b"]),
        model_name=str(tied_data["model_name"]),
        parameter_name=str(tied_data["parameter_name"])
    )

### Dimensionless Parameter

def serialize_scalar_parameter(param: Parameter) -> dict[str, Any]:
    """Create a dictionary representation of a Parameter instance."""
    data = {
        "name": str(param.name),
        "value": float(param.value),
        "fixed": bool(param.fixed),
        "min": float(param.bounds[0]),
        "max": float(param.bounds[1]),
    }
    if param.tied:
        data |= serialize_tied(param.tied)
    return data


def deserialize_scalar_parameter(data: dict[str, Any]) -> dict[str, Any]:
    out = {
        "value": float(data["value"]),
        "fixed": bool(data["fixed"]),
        "bounds": (float(data["min"]), float(data["max"])),
        "tied": None,
    }
    if "tied" in data:
        out["tied"] = deserialize_tied(data["tied"])
    return out

### Dimensional Parameter

def serialize_parameter(param: Parameter, unit: str | None) -> dict[str, Any]:
    out = serialize_scalar_parameter(param)
    if unit is None:
        return out
    out["unit"] = unit
    return out

def deserialize_parameter(data: dict[str, Any], unit: str | None) -> dict[str, Any]:
    if unit is None:
        return deserialize_scalar_parameter(data)
    
    curr_unit = Unit(data["unit"])
    def transform(val: float | None) -> float | None:
        if val is None:
            return None
        val = float(val)
        return (val * curr_unit).to(unit).value    

    out = {
        "value": transform(data["value"]),
        "fixed": data["fixed"],
        "bounds": (transform(data["min"]), transform(data["max"])),
        "tied": None,
    }
    if "tied" in data:
        out["tied"] = deserialize_tied(data["tied"])
    return out