from collections.abc import Callable

from numpy import float64, zeros
from numpy.typing import NDArray

from . import evaluate, fit_deriv


class _PowerLawBase:
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        raise NotImplementedError

    def __init__(
        self,
        func_name: str | None = None,
    ) -> None:
        self.func_name: str | None = func_name
        self.__wrapped__: Callable | None = self._get_wrapped(func_name)

    def __getstate__(self) -> dict:
        return {
            "func_name": self.func_name,
        }

    def __setstate__(self, state: dict) -> None:
        self.func_name = state["func_name"]
        self.__wrapped__ = self._get_wrapped(self.func_name)


###


class PowerLawEvaluate(_PowerLawBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: NDArray[float64],
        flux: float,
        alpha: float,
        *,
        x0: float,
        y: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if y is None:
            y = zeros(x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(y, x, flux, alpha, x0)
        return y


class PowerLawInverse(_PowerLawBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        y: NDArray[float64],
        flux: float,
        alpha: float,
        *,
        x0: float,
        x: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if x is None:
            x = zeros(y.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(x, y, flux, alpha, x0)
        return x


class PowerLawFitDeriv(_PowerLawBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: NDArray[float64],
        flux: float,
        alpha: float,
        *,
        x0: float,
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        if derivs is None:
            derivs = zeros((2, x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(derivs, x, flux, alpha, x0)
        return derivs
