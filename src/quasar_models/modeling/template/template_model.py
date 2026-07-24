from abc import ABC, abstractmethod

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.interpolation import create_interp_matrix
from quasar_models.modeling.base_model import BaseModel

class TemplateModel(BaseModel, ABC):
    ### Abstract methods/properties ##

    @abstractmethod
    def evaluate(self, *args): ...

    @abstractmethod
    def fit_deriv(self, *args): ...
    
    ### Properties

    @property
    def _interpolation_matrices(self) -> dict[str, tuple[csr_matrix_, FloatVector]]:
        return self.meta.get('_interpolation_matrices', {})
    
    @_interpolation_matrices.setter
    def _interpolation_matrices(self, value: dict[str, tuple[csr_matrix_, FloatVector]]) -> None:
        self.meta['_interpolation_matrices'] = value

    @property
    def allow_interp_fitting(self) -> bool:
        return self.meta['allow_interp_fitting']
    
    @allow_interp_fitting.setter
    def allow_interp_fitting(self, value: bool) -> None:
        self.meta['allow_interp_fitting'] = value
    
    def _calculate_interpolation_matrices(self, x_out: FloatVector) -> None:
        self._interpolation_matrices['interpolation_matrix'] = \
            create_interp_matrix.__wrapped__(
                self.template.x, x_out,
                left=0.0, right=0.0,
        )

    @abstractmethod
    def _choose_evaluate_func(self) -> None: 
        pass

    @abstractmethod
    def _choose_fit_deriv_func(self) -> None:
        pass

    def _prepare_model(self, x_out: FloatVector) -> None:
        super()._prepare_model()
        self._calculate_interpolation_matrices(x_out)

    def _unprepare_model(self) -> None:
        super()._unprepare_model()
        self._interpolation_matrices.clear()