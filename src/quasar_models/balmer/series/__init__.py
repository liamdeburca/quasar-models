import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmer_series_template": ["BalmerSeriesTemplate"],
    "io": [
        "PATH_TO_CACHE",
        "load",
        "load_from_cache",
        "save",
        "save_to_cache",
        "convert_path",
        "convert_params_to_path"
    ],
}

if TYPE_CHECKING:
    from .balmer_series_template import (
        BalmerSeriesTemplate,  # noqa: F401
    )
    from .io import (
        PATH_TO_CACHE,  # noqa: F401
        convert_params_to_path,  # noqa: F401
        convert_path,  # noqa: F401
        load,  # noqa: F401
        load_from_cache,  # noqa: F401
        save,  # noqa: F401
        save_to_cache,  # noqa: F401
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )