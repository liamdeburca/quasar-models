"""
Abstract model: abstract base class used for inheritance by all custom 
Astropy-compatible models.
"""
from abc import ABC, abstractmethod
from typing import Self, Union, ClassVar, Literal
from astropy.modeling.core import Fittable1DModel, _ModelMeta

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

class BaseModelMeta(_ModelMeta):
    def __or__(cls, other):
        return Union[cls, other]
    
    def __ror__(cls, other):
        return Union[other, cls]

class BaseModel(ABC, Fittable1DModel, metaclass=BaseModelMeta):
    model_type: ClassVar[Literal['pl', 'fe', 'ba', 'hg', 'em']]

    @property
    def pure_name(self) -> str:
        return self.name.split('#')[0]
    
    ### Fixed parameters
    
    @property
    def fixed_dict(self) -> dict[str, bool] | None:
        return self.meta.get('fixed_dict', None)
    
    @fixed_dict.setter
    def fixed_dict(self, value: dict[str, bool]) -> None:   
        self.meta['fixed_dict'] = value

    @fixed_dict.deleter
    def fixed_dict(self) -> None:
        self.meta.pop('fixed_dict', None)

    ### Tied parameters

    @property
    def tied_dict(self) -> dict[str, str] | None:
        return self.meta.get('tied', None)
    
    @tied_dict.setter
    def tied_dict(self, value: dict[str, str]) -> None:
        self.meta['tied'] = value

    @tied_dict.deleter
    def tied_dict(self) -> None:
        self.meta.pop('tied', None)
    
    ###

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected {cls.__name__} instance, "\
                f"got {type(value).__name__}"
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    def __lt__(self, other: Self) -> bool:
        return self.sorting_key < other.sorting_key
    
    def __gt__(self, other: Self) -> bool:
        return self.sorting_key > other.sorting_key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) \
            and all(
                getattr(self, pname) == getattr(other, pname)
                for pname in self.param_names
            ) \
            and self.name == other.name \
            and self.meta == other.meta
            
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
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """
        pass

    ### Serialisation

    # def __getstate__(self) -> dict:
    #     state = {
    #         param_name: getattr(self, param_name)
    #         for param_name in self.param_names
    #     }
    #     state['meta'] = self.meta
    #     state['name'] = self.name
    #     return state
    
    # def __setstate__(self, state: dict) -> None:
    #     self.__init__(
    #         *[state[param_name].value for param_name in self.param_names],
    #         name=state['name'],
    #         meta=state['meta'],
    #     )
    #     for param_name in self.param_names:
    #         param = state[param_name]
    #         curr_param = getattr(self, param_name)

    #         curr_param.value = param.value
    #         curr_param.fixed = param.fixed
    #         curr_param.bounds = param.bounds
    #         curr_param.tied = param.tied