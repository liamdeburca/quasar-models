import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "base_model": ["BaseModel"],
    "fitting": ["Fitter", "MCMCFitter"],
    "prepare_model": ["PrepareModel"],
    "sequential_model": ["SequentialModel"],
    "template": ["BaseTemplate", "TemplateModel"],
    "utils": ["LinearTie", "IdenticalTie"],
}

if TYPE_CHECKING:
    from .base_model import BaseModel  # noqa: F401
    from .fitting import Fitter, MCMCFitter  # noqa: F401
    from .prepare_model import PrepareModel  # noqa: F401
    from .sequential_model import SequentialModel  # noqa: F401
    from .template import BaseTemplate, TemplateModel  # noqa: F401
    from .utils import IdenticalTie, LinearTie  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )