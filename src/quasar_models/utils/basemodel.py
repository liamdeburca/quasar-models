"""
Abstract model: abstract base class used for inheritance by all custom 
Astropy-compatible models.
"""
from abc import ABC, abstractmethod
from typing import Self
from astropy.modeling import Fittable1DModel

class BaseModel(ABC, Fittable1DModel):
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

    @classmethod
    @abstractmethod
    def _validate(cls, value: object) -> Self:
        """
        Validate the input parameters for the model.
        """
        pass

    @classmethod
    @abstractmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """
        Define the pydantic core schema for model validation.
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