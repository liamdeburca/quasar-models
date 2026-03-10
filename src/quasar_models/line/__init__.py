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

class VProfileCopyDict:
    """
    Lorem ipsum.

    Methods
    -------
    __class_getitem__(n: int)
        Returns the corresponding VProfileCopy class for n profiles.

    Raises
    ------
    NotImplementedError
        If n is not in {1, 2, 3, 4, 5}.

    Notes
    -----
    Lorem ipsum.
    
    Examples
    --------
    >>> VProfileCopyDict[1]  # Returns VProfileCopy1G
    >>> VProfileCopyDict[3]  # Returns VProfileCopy3G
    """

    def __class_getitem__(cls, n: int) -> _VProfileCopy:
        match n:
            case 1: return VProfileCopy1G
            case 2: return VProfileCopy2G
            case 3: return VProfileCopy3G 
            case 4: return VProfileCopy4G
            case 5: return VProfileCopy5G
            case _: raise NotImplementedError(f"{n=} is not yet supported!")