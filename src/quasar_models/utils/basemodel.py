"""
Abstract model: abstract base class used for inheritance by all custom 
Astropy-compatible models.
"""
from abc import ABC, abstractmethod
from typing import Self, Union
from astropy.modeling.core import Fittable1DModel, _ModelMeta

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

class BaseModelMeta(_ModelMeta):
    def __or__(cls, other):
        return Union[cls, other]
    
    def __ror__(cls, other):
        return Union[other, cls]

class BaseModel(ABC, Fittable1DModel, metaclass=BaseModelMeta):
    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected {cls.__name__} instance, \
                got {type(value).__name__}"
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    def __lt__(self, other: Self) -> bool:
        return self.sorting_key < other.sorting_key
    
    def __gt__(self, other: Self) -> bool:
        return self.sorting_key > other.sorting_key
            
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model at given input values.
        """
        pass

    @abstractmethod
    def fit_deriv(self, *args, **kwargs):
        """
        Calculate the partial derivatives of the fitting function.
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Returns a string-representation of the model type.
        """
        pass

    @property
    @abstractmethod
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        pass