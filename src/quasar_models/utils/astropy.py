__all__ = [
    "apply_bounds",
    "get_configuration",
    "order_submodels",
    "separate_submodels",
]

from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import product

from astropy.modeling.core import Fittable1DModel
from numpy import clip, inf
from quasar_typing.astropy import CompoundModel_, Fittable1DModel_, Model_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.misc.literals import FluxComponent
from quasar_typing.numpy import FloatArray
from quasar_utils.decorators import validate_call


@validate_call
def apply_bounds(
    val: float | FloatArray,
    bounds: AstropyBounds,
) -> float | FloatArray:
    return clip(
        val,
        a_min=bounds[0] if (bounds[0] is not None) else -inf,
        a_max=bounds[1] if (bounds[1] is not None) else inf,
    )


@validate_call
def get_largest_possible_bounds(
    bounds: list[AstropyBounds],
    min_val: float = -inf,
    max_val: float = inf,
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

    def _lower(b: float | None) -> float:
        return min_val if b is None else b

    def _upper(b: float | None) -> float:
        return max_val if b is None else b
    
    lb: float = _lower(bounds[0][0])
    ub: float = _upper(bounds[0][1])

    for i in range(1, len(bounds)):
        combs = [
            _lower(a) * _upper(b) 
            for a, b in product((lb, ub), bounds[i])
        ]
        lb = min(combs)
        ub = max(combs)

    return lb, ub


@validate_call
def order_submodels(
    submodels: Fittable1DModel_ | Iterable[Fittable1DModel_],
    combine: bool = True,
) -> CompoundModel_ | list[Fittable1DModel_]:
    """
    ** PYDANTIC VALIDATED FUNCTION **
    """
    if isinstance(submodels, Fittable1DModel):
        return submodels if combine else [submodels]

    ms = sorted(submodels, key=lambda m: m.sorting_key)
    return sum(ms[1:], start=ms[0]) if combine else ms


@validate_call
def separate_submodels(
    submodels: Model_ | Iterable[Fittable1DModel_],
    combine: bool = True,
) -> dict[FluxComponent, Model_ | list[Fittable1DModel_] | None]:
    """
    ** PYDANTIC VALIDATED FUNCTION **

    Separates a collection of Astropy models based on their respective
    model_type properties:
    - 'pl': Power-law component
    - 'fe': Iron pseudo-continuum component
    - 'ba': Balmer pseudo-continuum component
    - 'hg': Host galaxy component
    - 'em': Emission line component
    """
    submodels_dict = defaultdict(list)
    for submodel in order_submodels(submodels, combine=False):
        submodels_dict[submodel.model_type].append(submodel)

    return {
        key: sum(ms[1:], start=ms[0]) if combine else ms
        for key, ms in submodels_dict.items()
    }


@validate_call
def get_configuration(model: Model_) -> dict[float, int]:
    """
    Retrieves the configuration of a given Astropy model, defined as the count
    of submodels corresponding to each unique wavelength parameter value.
    """
    ms = (model,) if model.n_submodels == 1 else model
    return Counter(m.wave.value for m in ms)


@validate_call
def get_model_parts(model: Model_) -> dict[FluxComponent, Model_ | None]:
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


@validate_call
def get_free_params(model: Model_) -> dict[str, bool]:
    return {
        p: not (model.fixed[p] or model.tied[p])
        for p in model.param_names
    }
