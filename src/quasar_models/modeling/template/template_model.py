from abc import ABC, abstractmethod
from typing import ClassVar

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.interpolation import create_interp_matrix

from quasar_models._core.modeling.template.cytemplate import CyTemplate
from quasar_models._core.modeling.utils import (
    _TemplateEvaluate,
    _TemplateFitDeriv,
)

from ..base_model import BaseModel
from .base_template import BaseTemplate


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
    def interpolation_matrix(
        self,
    ) -> tuple[csr_matrix_, FloatVector] | None:
        return self.meta.get("interpolation_matrix", None)

    @interpolation_matrix.setter
    def interpolation_matrix(
        self, 
        value: tuple[csr_matrix_, FloatVector] | None
    ) -> None:
        self.meta["interpolation_matrix"] = value

    @interpolation_matrix.deleter
    def interpolation_matrix(self) -> None:
        self.meta.pop("interpolation_matrix", None)

    def _set_interpolation_matrices(
        self, 
        x_out: FloatVector | None = None,
    ) -> None:
        """
        Precalculates the interpolation matrices. If no output grid is provided,
        interpolation is assumed to be unnecessary, i.e. the template is already
        on the desired grid.
        """
        if x_out is None:
            self.interpolation_matrix = None
        else:
            self.interpolation_matrix = create_interp_matrix.__wrapped__(
                getattr(self, self.TEMPLATE_KEYS[0]).x,
                x_out,
                left=0.0,
                right=0.0,
            )

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

    def _set_cytemplates(
        self, 
        fwhm: float | None = None,
        x_out: FloatVector | None = None,
    ) -> None:
        for k, _k in zip(self.CYTEMPLATE_KEYS, self.TEMPLATE_KEYS):
            template: BaseTemplate = getattr(self, _k)

            if fwhm is not None:
                # Create simple, 1d template
                assert x_out is not None
                template = template \
                    .create1DTemplate(fwhm) \
                    .interpolate(x_out)

            self.meta[k] = CyTemplate.fromTemplate(
                template,
                simplify=self.evaluate_func.simplify,
            )

    @abstractmethod
    def _set_kwargs(self) -> None:
        ...

    # Interp fitting

    @property
    def allow_interp_fitting(self) -> bool:
        return bool(self.meta["allow_interp_fitting"])

    @allow_interp_fitting.setter
    def allow_interp_fitting(self, value: bool) -> None:
        self.meta["allow_interp_fitting"] = bool(value)

    @property
    @abstractmethod
    def evaluate_func(self) -> _TemplateEvaluate: ...

    @abstractmethod
    def _choose_evaluate_func(self) -> None:
        pass

    @property
    @abstractmethod
    def fit_deriv_func(self) -> _TemplateFitDeriv: ...

    @abstractmethod
    def _choose_fit_deriv_func(self) -> None:
        pass

    def prepare_model(
        self, 
        x_out: FloatVector,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self._choose_evaluate_func(fixed=fixed)
        self._choose_fit_deriv_func(fixed=fixed)
        self._set_kwargs()
        
        if self.evaluate_func.rescaling:
            self._set_cytemplates(
                fwhm=self.fwhm.value,
                x_out=x_out,
            )
            self._set_interpolation_matrices()
        else:
            self._set_cytemplates()
            self._set_interpolation_matrices(
                x_out=x_out,
            )
        self._set_kwargs()

    def unprepare_model(self) -> None:
        super().unprepare_model()
        del self.interpolation_matrix
        del self.cytemplates
