__all__ = [
    "BalmerModel",
    "GaussianModel",
    "HostGalaxyModel",
    "IronModel",
    "PowerLawModel",
    "VProfileCopy",
]
from typing import Union

from .balmer import BalmerModel
from .continuum import PowerLawModel
from .host import HostGalaxyModel
from .iron import IronModel
from .line import GaussianModel, VProfileCopy

### Types

BaModel = BalmerModel
PlModel = PowerLawModel
HgModel = HostGalaxyModel
FeModel = IronModel
EmModel = Union[GaussianModel, VProfileCopy]