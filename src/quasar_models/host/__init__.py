import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "host_galaxy_model": ["HostGalaxyModel", "HgModel"],
    "host_galaxy_template": ["HostGalaxyTemplate"],
    "io": [
        "PATH_TO_CACHE",
        "PATH_TO_DATA",
    ]
}

if TYPE_CHECKING:
    from .host_galaxy_model import HgModel, HostGalaxyModel  # noqa: F401
    from .host_galaxy_template import HostGalaxyTemplate  # noqa: F401
    from .io import PATH_TO_CACHE, PATH_TO_DATA  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )