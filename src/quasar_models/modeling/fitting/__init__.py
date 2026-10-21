import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "emcee": ["MCMCFitter"],
    "nonlinear": ["Fitter"],
}

if TYPE_CHECKING:
    from .emcee import MCMCFitter  # noqa: F401
    from .nonlinear import Fitter  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )