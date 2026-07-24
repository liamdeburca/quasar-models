from typing import Callable
from numpy import float64, zeros
from numpy.typing import NDArray

from . import evaluate, fit_deriv

class _VProfileCopyBase:
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

class VProfileCopyEvaluate(_VProfileCopyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: NDArray[float64],
        strength_scale: float,
        strengths: NDArray[float64],
        fwhm_vs: NDArray[float64],
        v_offs: NDArray[float64],
        *,
        wave: float,
        sigma_res: float,
        y: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if y is None:
            y = zeros(x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(y, x, strength_scale, strengths, fwhm_vs, v_offs, wave, sigma_res)
        return y
    
class VProfileCopyFitDeriv(_VProfileCopyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None
    
    def __call__(
        self,
        x: NDArray[float64],
        strength_scale: float,
        strengths: NDArray[float64],
        fwhm_vs: NDArray[float64],
        v_offs: NDArray[float64],
        *,
        wave: float,
        sigma_res: float,
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if derivs is None:
            derivs = zeros((1 + 3 * strengths.size, x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(derivs, x, strength_scale, strengths, fwhm_vs, v_offs, wave, sigma_res)
        return derivs