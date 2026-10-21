import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "gaussian": ["GaussianModel"],
    "line": [
        "create_vprofilecopy",
        "rename_gaussians",
        "EmModel",
    ],
    "vprofilecopies._vprofilecopy": ["_VProfileCopy"],
    "vprofilecopies.vprofilecopies": [
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
    from .gaussian import GaussianModel  # noqa: F401
    from .line import (  # noqa: F401
        EmModel,
        create_vprofilecopy,
        rename_gaussians,
    )
    from .vprofilecopies.vprofilecopies import (  # noqa: F401
        VProfileCopy,
        VProfileCopy1G,
        VProfileCopy2G,
        VProfileCopy3G,
        VProfileCopy4G,
        VProfileCopy5G,
        VProfileCopyDict,
        _VProfileCopy,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )