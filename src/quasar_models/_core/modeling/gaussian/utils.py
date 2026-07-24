from typing import Callable
from numpy import float64, zeros
from numpy.typing import NDArray
from . import evaluate, fit_deriv

class _GaussianBase:
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

###

class GaussianEvaluate(_GaussianBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: NDArray[float64],
        strength: float,
        fwhm_v: float,
        v_off: float,
        *,
        wave: float,
        sigma_res: float,
        y: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if y is None:
            y = zeros(x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(y, x, strength, fwhm_v, v_off, wave, sigma_res)
        return y

class GaussianFitDeriv(_GaussianBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None
    
    def __call__(
        self,
        x: NDArray[float64],
        strength: float,
        fwhm_v: float,
        v_off: float,
        *,
        wave: float,
        sigma_res: float,
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if derivs is None:
            derivs = zeros((3, x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(derivs, x, strength, fwhm_v, v_off, wave, sigma_res)
        return derivs