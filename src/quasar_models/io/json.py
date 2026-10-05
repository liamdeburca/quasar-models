"""JSON file I/O helpers for model serialization.

These are thin convenience wrappers around the serialization helpers in
``src.quasar_models.io.serialization``. They provide file-based read/write
operations for single models or compound (additive) models.
"""
from pathlib import Path
import json
from typing import Any

from quasar_utils.setup import Info

from quasar_models.modeling.base_model import BaseModel
from .serialization.compound import (
    serialize_compound_model,
    deserialize_compound_model,
)


def write_model_json(path: str | Path, model: BaseModel, info: Info, *, indent: int = 2) -> Path:
    """Serialize a model (or additive compound model) and write to a JSON file.

    Returns the Path to the written file.
    """
    p = Path(path)
    data = serialize_compound_model(model, info)
    with p.open("w") as f:
        json.dump(data, f, indent=indent)
    return p


def read_model_json(path: str | Path, info: Info) -> BaseModel:
    """Read a model JSON file and deserialize into a model instance.

    Returns the reconstructed model (single or compound).
    """
    p = Path(path)
    with p.open("r") as f:
        data: dict[str, Any] = json.load(f)
    return deserialize_compound_model(data, info)
