import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmer": [
        "BalmerModel", 
        "BaModel",
    ],
    "continuum": [
        "PlModel", 
        "PowerLawModel",
    ],
    "host": [
        "HgModel", 
        "HostGalaxyModel",
    ],
    "iron": [
        "FeModel", 
        "IronModel",
    ],
    "line": [
        "EmModel", 
        "GaussianModel", 
        "VProfileCopy",
    ],
}

if TYPE_CHECKING:
    from .balmer import BalmerModel, BaModel  # noqa: F401
    from .continuum import PlModel, PowerLawModel  # noqa: F401
    from .host import HgModel, HostGalaxyModel  # noqa: F401
    from .iron import FeModel, IronModel  # noqa: F401
    from .line import EmModel, GaussianModel, VProfileCopy  # noqa: F401

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
