from collections.abc import Iterable
from typing import Any, Union

from quasar_typing.astropy import CompoundModel_
from quasar_utils.setup import Info

from quasar_models import (
    BalmerModel,
    GaussianModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.line import (
    VProfileCopy1G,
    VProfileCopy2G,
    VProfileCopy3G,
    VProfileCopy4G,
    VProfileCopy5G,
)
from quasar_models.modeling.base_model import BaseModel


def serialize_compound_model(
    model: Union[BaseModel, CompoundModel_[BaseModel]], 
    info: Info,
) -> dict[str, Any]:
    """Serialize a single model or an additive compound model.

    If `model` has `n_submodels` > 1 it is treated as a compound model and
    each submodel is serialized using its own `serialize(info)` method. The
    returned dict is a mapping of keys produced by the underlying serializations
    (i.e. "ClassName::model_name" -> data). For consistency with other
    serializers the function returns a single dict containing all submodel
    entries (or a single entry for non-compound models).
    """
    out: dict[str, Any] = {}
    ms: Iterable[BaseModel] = (model,) if model.n_submodels == 1 else model
    for m in sorted(ms, key=lambda m: m.sorting_key):
        out.update(m.serialize(info))
    return out


def deserialize_compound_model(data: dict[str, Any], info: Info) -> BaseModel:
    """Deserialize a serialized model dict into a single model or compound model.

    The input `data` is expected to be the dict produced by
    `serialize_compound_model` (i.e. keys of the form "ClassName::model_name"
    mapping to per-model dicts). The returned object is a single model if
    `data` contains a single entry; otherwise an additive compound model is
    constructed by deserializing each submodel and summing them. Submodels are
    sorted using their `sorting_key` before being combined.
    """
    items = list(data.items())
    if len(items) == 0:
        raise ValueError("No model data provided for deserialization")

    models: list[BaseModel] = []
    for key, val in items:
        try:
            cls_: type[BaseModel] = {
                "BalmerModel": BalmerModel,
                "PowerLawModel": PowerLawModel,
                "IronModel": IronModel,
                "HostGalaxyModel": HostGalaxyModel,
                "GaussianModel": GaussianModel,
                "VProfileCopy1G": VProfileCopy1G,
                "VProfileCopy2G": VProfileCopy2G,
                "VProfileCopy3G": VProfileCopy3G,
                "VProfileCopy4G": VProfileCopy4G,
                "VProfileCopy5G": VProfileCopy5G,
            }[key]
        except KeyError as exc:
            raise KeyError(f"Unknown model class: {key}") from exc
        
        models.append(cls_.deserialize(val, info))

    if len(models) == 1:
        return models[0]

    models = sorted(models, key=lambda m: m.sorting_key)
    return sum(models[1:], start=models[0])