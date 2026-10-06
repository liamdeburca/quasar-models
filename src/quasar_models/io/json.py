"""JSON file I/O helpers for model serialization.

These are thin convenience wrappers around the serialization helpers in
``src.quasar_models.io.serialization``. They provide file-based read/write
operations for single models or compound (additive) models.
"""
from importlib import import_module
from pathlib import Path
from typing import Any

from quasar_utils.setup import Info

from quasar_models.modeling.base_model import BaseModel

from .serialization.compound import (
    deserialize_compound_model,
    serialize_compound_model,
)


def _load_pkg(suffix: str):
    match suffix:
        case ".json":
            pkg_name = "json"
        case ".yaml" | ".yml":
            pkg_name = "yaml"
        case ".toml":
            pkg_name = "toml"
        case ".toon":
            pkg_name = "toons"
        case _:
            raise ValueError(f"Unsupported file extension: {suffix}")

    try:
        return import_module(pkg_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import package '{pkg_name}' for suffix '{suffix}'"
        ) from e

def model_to_file(
    path: str | Path, 
    model: BaseModel, 
    info: Info, 
    **kwargs: dict[str, Any],
) -> Path:
    """Serialize a model (or additive compound model) and write to a file.

    Returns the Path to the written file.
    """
    path = Path(path).resolve()
    data = serialize_compound_model(model, info)
    suffix = path.suffix.lower()
    pkg = _load_pkg(suffix)

    if suffix == ".toml":
        kwargs.clear()
    else:
        kwargs = {"indent": 2} | kwargs

    with path.open("w") as f:
        pkg.dump(data, f, **kwargs)

    return path


def file_to_model(path: str | Path, info: Info) -> BaseModel:
    """Read a model file and deserialize into a model instance.

    Returns the reconstructed model (single or compound).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    pkg = _load_pkg(suffix)

    if suffix in {".yaml", ".yml"}:
        args = (pkg.Loader,)
    else:
        args = ()
    
    with path.open("r") as f:
        data = pkg.load(f, *args)
        
    return deserialize_compound_model(data, info)
