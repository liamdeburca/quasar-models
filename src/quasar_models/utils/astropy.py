__all__ = [
    'apply_bounds',
    'order_submodels',
    'separate_submodels',
    'get_configuration',
]

from typing import Iterable
from collections import defaultdict, Counter
from numpy import clip, inf
from astropy.modeling.core import Fittable1DModel
from pydantic import validate_call

from quasar_typing.numpy import FloatArray
from quasar_typing.bounds import AstropyBounds
from quasar_typing.astropy import Model_, Fittable1DModel_
from quasar_typing.misc.literals import FluxComponent_

@validate_call(validate_return=False)
def apply_bounds(
    val: float | FloatArray,
    bounds: AstropyBounds,
) -> float | FloatArray:
    return clip(
        val,
        a_min = bounds[0] if (bounds[0] is not None) else -inf,
        a_max = bounds[1] if (bounds[1] is not None) else inf,
    )

@validate_call(validate_return=False)
def order_submodels(
    submodels: Model_ | Iterable[Fittable1DModel_],
    combine: bool = True,
) -> Model_ | list[Fittable1DModel_]:
    """
    Lorem ipsum...
    """
    if isinstance(submodels, Fittable1DModel):
        return submodels if combine else [submodels]
    
    _submodels: list[Fittable1DModel] = sorted(
        submodels, 
        key=lambda m: m.sorting_key,
    )
    if combine: return sum(_submodels[1:], start=_submodels[0])
    else:       return _submodels

@validate_call(validate_return=False)
def separate_submodels(
    submodels: Model_ | Iterable[Fittable1DModel_],
    combine: bool = True,
) -> dict[
    FluxComponent_,
    Model_ | list[Fittable1DModel_] | None,
]:
    """
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

    out = {key: submodels for key, submodels in submodels_dict.items()}
    if combine:
        for key, submodels in out.items():
            out[key] = sum(submodels[1:], start=submodels[0])

    return out

@validate_call(validate_return=False)
def get_configuration(
    model: Model_,
) -> dict[float, int]:
    """
    Retrieves the configuration of a given Astropy model, defined as the count
    of submodels corresponding to each unique wavelength parameter value.
    """
    if model.n_submodels == 1: submodels = [model]
    else:                      submodels = model
    
    return Counter(m.wave.value for m in submodels)

@validate_call(validate_return=False)
def get_model_parts(
    model: Model_,
) -> dict[FluxComponent_, Model_ | None]:
    """
    Lorem ipsum.
    """
    parts = dict(
        pl = None,
        fe = None,
        ba = None,
        hg = None,
        em = None,
    )
    submodels = model if (model.n_submodels > 1) else [model]

    for submodel in submodels:
        key = submodel.model_type

        if parts[key] == None: parts[key] = submodel
        else:                  parts[key] += submodel

    return parts

@validate_call(validate_return=False)
def get_free_params(model: Model_) -> dict[str, bool]:
    return dict([
        (p, not (model.fixed[p] or model.tied[p])) \
        for p in model.param_names
    ])