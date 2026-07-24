from typing import Callable
from numpy import float64, zeros, add
from numpy.typing import NDArray

from ..utils import _interp, _interp2d, _interp_matrix, _interp2d_matrix
from . import evaluate, fit_deriv

class _TemplateBase:
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable:
        raise NotImplementedError

    def __init__(
        self,
        func_name: str | None = None,
    ) -> None:
        self.func_name: str | None = func_name
        self.__wrapped__: Callable | None = self._get_wrapped(func_name)

    def __getstate__(self) -> dict:
        return {
            'func_name': self.func_name,
        }
    
    def __setstate__(self, state: dict) -> None:
        self.func_name = state['func_name']
        self.__wrapped__ = self._get_wrapped(self.func_name)

class TemplateEvaluate(_TemplateBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: NDArray[float64],
        flux: float,
        fwhm: float,
        *,
        template_fwhm: NDArray[float64],
        template_x: NDArray[float64],
        template_data: NDArray[float64],
        sigma_res: float,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
        y: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        _y = zeros(template_x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _y,
                flux, fwhm, 
                template_fwhm, template_x, template_data,
                sigma_res, n_scales,
            )
            if interpolation_matrix is None:
                _y = _interp(x, template_x, _y)
            else:
                _y = _interp_matrix(_y, interpolation_matrix)
        
        return _y if y is None else add(y, _y, out=y)
    
class TemplateFitDeriv(_TemplateBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: NDArray[float64],
        flux: float,
        fwhm: float,
        *,
        template_fwhm: NDArray[float64],
        template_x: NDArray[float64],
        template_data: NDArray[float64],
        sigma_res: float,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        _derivs = zeros((3, template_x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _derivs,
                flux, fwhm, 
                template_fwhm, template_x, template_data,
                sigma_res, n_scales,
            )
            if interpolation_matrix is None:
                _derivs = _interp2d(x, template_x, _derivs)
            else:
                _derivs = _interp2d_matrix(_derivs, interpolation_matrix)
        
        return _derivs if derivs is None else add(derivs, _derivs, out=derivs)