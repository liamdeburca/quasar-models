from typing import TypeVar, Iterable
from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import Model_

from .basemodel import BaseModel
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
        self.model: M = model.copy() if copy else model
        self.copy: bool = copy

    @classmethod
    def prepare_submodels(
        cls,
        x: FloatVector,
        submodels: Iterable[BaseModel],
    ) -> None:
        for submodel in submodels:
            assert 'fixed_dict' not in submodel.meta
            submodel.meta['fixed_dict'] = dict(submodel.fixed)
            if isinstance(submodel, TemplateModel):
                submodel._calculate_interpolation_matrices(x)

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
            submodel.meta.pop('fixed_dict', None)
            if isinstance(submodel, TemplateModel):
                submodel._interpolation_matrices.clear()

    @classmethod
    def unprepare_model(
        cls, 
        model: M,
    ) -> None:
        submodels = (model,) if model.n_submodels == 1 else model
        cls.unprepare_submodels(submodels)

    def __enter__(self) -> M | None:
        self.prepare_model(self.x, self.model)
        if self.copy:
            return self.model

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.unprepare_model(self.model)