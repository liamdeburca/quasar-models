from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.interpolation import create_interp_matrix

from quasar_models._core.modeling.template.cytemplate import CyTemplate
from quasar_models.modeling.base_model import BaseModel


class TemplateModel(BaseModel, ABC):
    TEMPLATE_KEYS: ClassVar[tuple[str, ...]] = ("template",)
    CYTEMPLATE_KEYS: ClassVar[tuple[str, ...]] = ("cytemplate",)

    ### Abstract methods/properties ##

    @abstractmethod
    def evaluate(self, *args): ...

    @abstractmethod
    def fit_deriv(self, *args): ...

    ### Properties

    # Interpolation matrix

    @property
    def interpolation_matrix(self) -> tuple[csr_matrix_, FloatVector] | None:
        return self.meta.get("interpolation_matrix", None)

    @interpolation_matrix.setter
    def interpolation_matrix(
        self, value: tuple[csr_matrix_, FloatVector] | None
    ) -> None:
        self.meta["interpolation_matrix"] = value

    @interpolation_matrix.deleter
    def interpolation_matrix(self) -> None:
        self.meta.pop("interpolation_matrix", None)

    def _set_interpolation_matrices(self, x_out: FloatVector) -> None:
        self.interpolation_matrix = create_interp_matrix.__wrapped__(
            getattr(self, self.TEMPLATE_KEYS[0]).x,
            x_out,
            left=0.0,
            right=0.0,
        )

    def _del_interpolation_matrices(self) -> None:
        del self.interpolation_matrix

    # Cython Templates

    @property
    def cytemplates(self) -> dict[str, CyTemplate | None]:
        return {k: self.meta.get(k, None) for k in self.CYTEMPLATE_KEYS}

    @cytemplates.setter
    def cytemplates(self, value: dict[str, CyTemplate | None]) -> None:
        for k in self.CYTEMPLATE_KEYS:
            if k in value:
                self.meta[k] = value[k]

    @cytemplates.deleter
    def cytemplates(self) -> None:
        for k in self.CYTEMPLATE_KEYS:
            self.meta.pop(k, None)

    def _set_cytemplates(self) -> None:
        for k, _k in zip(self.CYTEMPLATE_KEYS, self.TEMPLATE_KEYS):
            self.meta[k] = CyTemplate.fromTemplate(
                getattr(self, _k),
                simplify=self.evaluate_func.simplify,
            )

    def _del_cytemplates(self) -> None:
        del self.cytemplates

    # Interp fitting

    @property
    def allow_interp_fitting(self) -> bool:
        return bool(self.meta["allow_interp_fitting"])

    @allow_interp_fitting.setter
    def allow_interp_fitting(self, value: bool) -> None:
        self.meta["allow_interp_fitting"] = bool(value)

    @abstractmethod
    def _choose_evaluate_func(self) -> None:
        pass

    @abstractmethod
    def _choose_fit_deriv_func(self) -> None:
        pass

    def prepare_model(self, x_out: FloatVector) -> None:
        super().prepare_model()
        self._set_interpolation_matrices(x_out)
        self._set_cytemplates()

    def unprepare_model(self) -> None:
        super().unprepare_model()
        self._del_interpolation_matrices()
        self._del_cytemplates()
