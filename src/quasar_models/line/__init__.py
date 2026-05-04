"""
    Lorem ipsum.
"""

__all__ = [
    'GaussianModel',
    'VProfileCopy1G',
    'VProfileCopy2G',
    'VProfileCopy3G',
    'VProfileCopy4G',
    'VProfileCopy5G',
    'VProfileCopyDict',
]

from .gaussian import GaussianModel
from .vprofilecopies import (
    _VProfileCopy,
    VProfileCopy1G,
    VProfileCopy2G,
    VProfileCopy3G,
    VProfileCopy4G,
    VProfileCopy5G,
)

VProfileCopyDict: dict[int, _VProfileCopy] = {
    1: VProfileCopy1G,
    2: VProfileCopy2G,
    3: VProfileCopy3G,
    4: VProfileCopy4G,
    5: VProfileCopy5G,
}