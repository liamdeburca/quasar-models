import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "_vprofilecopy": ["_VProfileCopy"],
    "vprofilecopies": [
        "VProfileCopy",
        "VProfileCopy1G",
        "VProfileCopy2G",
        "VProfileCopy3G",
        "VProfileCopy4G",
        "VProfileCopy5G",
        "VProfileCopyDict",
    ],
}

if TYPE_CHECKING:
    from ._vprofilecopy import _VProfileCopy  # noqa: F401
    from .vprofilecopies import (  # noqa: F401
        VProfileCopy,
        VProfileCopy1G,
        VProfileCopy2G,
        VProfileCopy3G,
        VProfileCopy4G,
        VProfileCopy5G,
        VProfileCopyDict,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )