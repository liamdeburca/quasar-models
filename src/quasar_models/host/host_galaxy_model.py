"""
AstroPy compatible model: HostGalaxyModel.
"""
from logging import getLogger
from astropy.modeling import Parameter
from ..utils.basemodel import BaseModel

logger = getLogger(__name__)

class HostGalaxyModel(BaseModel):
    """
    Notes
    -----
    Cannot be combined with pipe operator ("|") in type annotations! Use 
    typing.Union or typing.Optional instead.
    """
    flux = Parameter(min=0)
    fwhm = Parameter(min=0)