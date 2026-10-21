import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "iron_model": ["IronModel", "FeModel"],
    "iron_template": ["IronTemplate"],
    "io": [
        "PATH_TO_CACHE",
        "PATH_TO_DATA",
    ]
}

if TYPE_CHECKING:
    from .io import PATH_TO_CACHE, PATH_TO_DATA  # noqa: F401
    from .iron_model import FeModel, IronModel  # noqa: F401
    from .iron_template import IronTemplate  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )