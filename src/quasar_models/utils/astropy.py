import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import wraps
from itertools import product
from typing import Union

import pydantic
from numpy import clip
from quasar_typing.astropy.modeling import (
    CompoundModelType,
    Fittable1DModelType,
    ModelType,
)
from quasar_typing.bounds import AstropyBounds
from quasar_typing.literals import FluxComponent
from quasar_typing.numpy import FloatArray


@pydantic.validate_call
def apply_bounds(
    val: float | FloatArray,
    bounds: AstropyBounds,
) -> float | FloatArray:
    return _apply_bounds(val, bounds)


@pydantic.validate_call
def get_largest_possible_bounds(
    bounds: list[AstropyBounds], 
    min_val: float = -math.inf, 
    max_val: float = math.inf,
) -> AstropyBounds:
    """
    Return the largest possible bounds created as a product of the input bounds.

    If a lower bound is None, it is replaced with `min_val`. If an upper bound 
    is None, it is replaced with `max_val`.

    Example:
    ```
        get_largest_possible_bounds([(0, 2), (-5, 10)]) -> (-5, 20)
    ```
    """
    return _get_largest_possible_bounds(
        bounds, 
        min_val=min_val, 
        max_val=max_val,
    )


@pydantic.validate_call
def order_submodels(
    submodels: Union[Fittable1DModelType, Iterable[Fittable1DModelType]],
    combine: bool = True,
) -> Union[CompoundModelType, list[Fittable1DModelType]]:
    return _order_submodels(submodels, combine=combine)


@pydantic.validate_call
def separate_submodels(
    submodels: Union[ModelType, Iterable[Fittable1DModelType]],
    combine: bool = True,
) -> dict[FluxComponent, Union[ModelType, list[Fittable1DModelType], None]]:
    """
    Separates a collection of Astropy models based on their respective
    model_type properties:
    - 'pl': Power-law component
    - 'fe': Iron pseudo-continuum component
    - 'ba': Balmer pseudo-continuum component
    - 'hg': Host galaxy component
    - 'em': Emission line component
    """
    return _separate_submodels(submodels, combine=combine)


@pydantic.validate_call
def get_configuration(model: ModelType) -> dict[float, int]:
    """
    Retrieves the configuration of a given Astropy model, defined as the count
    of submodels corresponding to each unique wavelength parameter value.
    """
    return _get_configuration(model)


@pydantic.validate_call
def get_model_parts(model: ModelType) -> dict[FluxComponent, Union[ModelType, None]]:
    return _get_model_parts(model) 


@pydantic.validate_call
def get_free_params(model: ModelType) -> dict[str, bool]:
    return _get_free_params(model) 


### Private API

@wraps(apply_bounds)
def _apply_bounds(
    val: float | FloatArray,
    bounds: AstropyBounds,
) -> float | FloatArray:
    return clip(
        val,
        a_min=-math.inf if (bounds[0] is None) else bounds[0],
        a_max=math.inf if (bounds[1] is None) else bounds[1]
    )


@wraps(get_largest_possible_bounds)
def _get_largest_possible_bounds(
    bounds: list[AstropyBounds], 
    min_val: float = -math.inf, 
    max_val: float = math.inf,
) -> AstropyBounds:
    def _lower(b: Union[float, None]) -> float:
        return min_val if b is None else b

    def _upper(b: Union[float, None]) -> float:
        return max_val if b is None else b

    lb: float = _lower(bounds[0][0])
    ub: float = _upper(bounds[0][1])
    for i in range(1, len(bounds)):
        combs = [
            _lower(a) * _upper(b) for a, b in product((lb, ub), bounds[i])
        ]
        lb = min(combs)
        ub = max(combs)
    return lb, ub


@wraps(order_submodels)
def _order_submodels(
    submodels: Union[Fittable1DModelType, Iterable[Fittable1DModelType]],
    combine: bool = True,
) -> Union[CompoundModelType, list[Fittable1DModelType]]:
    if not isinstance(submodels, Iterable):
        return submodels if combine else [submodels]

    ms = sorted(submodels, key=lambda m: m.sorting_key)
    return sum(ms[1:], start=ms[0]) if combine else ms


@wraps(separate_submodels)
def _separate_submodels(
    submodels: Union[ModelType, Iterable[Fittable1DModelType]],
    combine: bool = True,
) -> dict[FluxComponent, Union[ModelType, list[Fittable1DModelType], None]]:
    submodels_dict = defaultdict(list)
    for submodel in _order_submodels(submodels, combine=False):
        submodels_dict[submodel.model_type].append(submodel)

    return {
        key: sum(ms[1:], start=ms[0]) if combine else ms
        for key, ms in submodels_dict.items()
    }


@wraps(get_configuration)
def _get_configuration(model: ModelType) -> dict[float, int]:
    ms = (model,) if model.n_submodels == 1 else model
    return Counter(m.wave.value for m in ms)


@wraps(get_model_parts)
def _get_model_parts(model: ModelType) -> dict[FluxComponent, Union[ModelType, None]]:
    parts = {
        "pl": None,
        "fe": None,
        "ba": None,
        "hg": None,
        "em": None,
    }
    submodels = model if (model.n_submodels > 1) else [model]

    for submodel in submodels:
        key = submodel.model_type
        if parts[key] == None:
            parts[key] = submodel
        else:
            parts[key] += submodel

    return parts


@wraps(get_free_params)
def _get_free_params(model: ModelType) -> dict[str, bool]:
    return {
        p: not (model.fixed[p] or model.tied[p]) 
        for p in model.param_names
    }