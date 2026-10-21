import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmer_model": ["BalmerModel", "BaModel"],
    "continuum": ["BalmerContinuumTemplate"],
    "series": ["BalmerSeriesTemplate"],
}

if TYPE_CHECKING:
    from .balmer_model import BalmerModel, BaModel  # noqa: F401
    from .continuum import BalmerContinuumTemplate  # noqa: F401
    from .series import BalmerSeriesTemplate  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )