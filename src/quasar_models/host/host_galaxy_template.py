from pathlib import Path
from ..utils.template import BaseTemplate

_this_file: Path = Path(__file__).resolve()

#Place cache in project root directory, in a hidden folder named '.cache/host_galaxy_templates'
PATH_TO_CACHE: Path = _this_file.parents[3] / '.cache/host_galaxy_templates'
if not PATH_TO_CACHE.exists(): PATH_TO_CACHE.mkdir(parents=True)

class HostGalaxyTemplate(BaseTemplate):
    """
    Template class specifically designed for Host Galaxy templates.
    """
    PATH_TO_CACHE: Path = PATH_TO_CACHE