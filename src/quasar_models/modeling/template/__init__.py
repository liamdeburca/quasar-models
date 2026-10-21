import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "base_template": ["BaseTemplate"],
    "template_model": ["TemplateModel"]
}

if TYPE_CHECKING:
    from .base_template import BaseTemplate  # noqa: F401
    from .template_model import TemplateModel  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )