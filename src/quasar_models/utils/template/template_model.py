from abc import ABC, abstractmethod
from numpy import ndarray
from cachetools import LRUCache

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

from quasar_utils.interpolation import create_interp_matrix

from ..basemodel import BaseModel

class TemplateModel(BaseModel, ABC):
    """
    Abstract base class for models that are based on templates.

    ** Inherited methods/properties **
    - __init__ (method)
    - __call__ (method): populates the interpolation matrix cache and calls the 
                         evaluate method.
    - _perform_interp_fitting (property)
    - _get_interpolation_matrices (method)
    - _calculate_interpolation_matrices (method)

    ** Abstract methods/properties **
    - evaluate (method)
    - fit_deriv (method)
    """
    ### Abstract methods/properties ##

    @abstractmethod
    def evaluate(self, *args): ...

    @abstractmethod
    def fit_deriv(self, *args): ...

    ### Inherited methods/properties ###

    def _initialise_cache(self, maxsize: int) -> None:
        if getattr(self, '_interpolation_cache', None) is None:
            self._interpolation_cache: LRUCache[
                int, 
                dict[str, tuple[csr_matrix_, FloatVector]],
            ] = LRUCache(maxsize=maxsize)
        else:
            self._interpolation_cache.clear()

    @property
    def _perform_interp_fitting(self) -> bool: 
        return self.allow_interp_fitting

    def __call__(self, x: FloatVector) -> FloatVector:
        if isinstance(x, ndarray):
            _ = self._calculate_interpolation_matrices(x)
        return self.evaluate(x, *self.parameters)

    def _get_interpolation_matrices(
        self, 
        x_out: FloatVector,
    ) -> dict[str, tuple[csr_matrix_, FloatVector]]:
        cache_key = hash(x_out.tobytes())
        return self._interpolation_cache[cache_key]
    
    def _calculate_interpolation_matrices(
        self,
        x_out: FloatVector,
    ) -> dict[str, tuple[csr_matrix_, FloatVector]]:
        cache_key = hash(x_out.tobytes())
        if cache_key not in self._interpolation_cache:
            self._interpolation_cache[cache_key] = {
                'interpolation_matrix': create_interp_matrix.__wrapped__(
                    self.template.x, x_out, 
                    left=0., right=0.,
                ),
            }
        return self._interpolation_cache[cache_key]
