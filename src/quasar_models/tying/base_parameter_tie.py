from abc import ABC, abstractmethod
from typing import Self
from astropy.modeling import CompoundModel, Fittable1DModel, Model

class BaseParameterTie(ABC):
    def __init__(
        self,
        target_name: str,
        target_parameter: str,
    ) -> None:
        self.target_name = target_name
        self.target_parameter = target_parameter

    @abstractmethod
    def __call__(self, model: CompoundModel) -> float: ...

    def __getstate__(self) -> dict:
        return {
            'target_name': self.target_name,
            'target_parameter': self.target_parameter,
        }
    
    def __setstate__(self, state: dict) -> None:
        self.__init__(
            target_name=state['target_name'],
            target_parameter=state['target_parameter'],
        )

    def is_applicable_on(self, model: Model) -> bool:
        """
        Checks whether the target model and parameter exist in the provided model.
        """
        if model.n_submodels == 1:
            return (self.target_name == model.name) \
                and (self.target_parameter in model.param_names)
        
        if not any(self.target_name == m.name for m in model):
            return False

        if self.target_parameter not in model[self.target_name].param_names:
            return False
        
        return True
    
    def assert_is_applicable_on(self, model: CompoundModel) -> None:
        if not self.is_applicable_on(model):
            msg = "Parameter tie is not applicable on the provided model."
            raise ValueError(msg)
    
    @classmethod
    def from_compound_model(
        cls, 
        compound_model: CompoundModel,
        target_name: str,
        target_parameter: str,
    ) -> Self:
        tie = cls(target_name, target_parameter)
        tie.assert_is_applicable_on(compound_model)
        return tie
    
    @classmethod
    def from_model(
        cls,
        model: Fittable1DModel,
        target_parameter: str,
    ) -> Self:
        tie = cls(model.name, target_parameter)
        tie.assert_is_applicable_on(model)
        return tie