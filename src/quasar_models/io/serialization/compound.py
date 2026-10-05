from typing import Any

from quasar_utils.setup import Info

from quasar_models.modeling.base_model import BaseModel
from quasar_models.utils.serialization import serialize_parameter, serialize_quantity


def serialize_compound_model(model: BaseModel, info: Info) -> dict[str, Any]:
    """Serialize a single model or an additive compound model.

    If `model` has `n_submodels` > 1 it is treated as a compound model and
    each submodel is serialized using its own `serialize(info)` method. The
    returned dict is a mapping of keys produced by the underlying serializations
    (i.e. "ClassName::model_name" -> data). For consistency with other
    serializers the function returns a single dict containing all submodel
    entries (or a single entry for non-compound models).
    """
    out: dict[str, Any] = {}
    if getattr(model, "n_submodels", 1) > 1:
        # Compound model: iterate submodels (they should be additive)
        for sub in sorted((m for m in model), key=lambda m: m.sorting_key):
            out.update(sub.serialize(info))
    else:
        out.update(model.serialize(info))
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
        cls_name, name = key.split("::", 1)
        # Locate the class from the global quasar_models package
        # We import lazily to avoid circular imports at module load time
        import importlib

        package = importlib.import_module("quasar_models")
        cls = getattr(package, cls_name)
        model = cls.deserialize(val, name, info)
        models.append(model)

    if len(models) == 1:
        return models[0]

    # Sort submodels by sorting_key
    models = sorted(models, key=lambda m: m.sorting_key)

    # Sum (add) models to create an additive compound model
    compound = models[0]
    for m in models[1:]:
        compound = compound + m
    return compound
