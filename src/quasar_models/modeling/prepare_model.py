from typing import TypeVar, Iterable, Optional
from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import Model_

from .sequential_model import SequentialModel
from .base_model import BaseModel
from .template.template_model import TemplateModel

M = TypeVar('M', bound=Model_)

class PrepareModel:
    def __init__(
        self,
        *,
        x: FloatVector,
        model: M,
        copy: bool = False,
    ) -> None:
        self.x: FloatVector = x
        self.model: SequentialModel = SequentialModel(
            model.copy() if copy else model
        )
        self.copy: bool = copy

    @classmethod
    def prepare_submodels(
        cls,
        x: FloatVector,
        submodels: Iterable[BaseModel],
    ) -> None:
        for submodel in submodels:
            if isinstance(submodel, TemplateModel):
                submodel._prepare_model(x)
            else:
                submodel._prepare_model()

    @classmethod
    def prepare_model(
        cls,
        x: FloatVector,
        model: M,
    ) -> None:
        submodels = (model,) if model.n_submodels == 1 else model
        cls.prepare_submodels(x, submodels)
    
    @classmethod
    def unprepare_submodels(
        cls, 
        submodels: Iterable[BaseModel],
    ) -> None:
        for submodel in submodels:
            submodel._unprepare_model()

    @classmethod
    def unprepare_model(
        cls, 
        model: M,
    ) -> None:
        submodels = (model,) if model.n_submodels == 1 else model
        cls.unprepare_submodels(submodels)

    def __enter__(self) -> Optional[SequentialModel]:
        self.prepare_submodels(self.x, self.model.submodels)
        if self.copy:
            return self.model

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.unprepare_submodels(self.model.submodels)
