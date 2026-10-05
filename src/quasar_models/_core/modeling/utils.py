from collections.abc import Callable
from typing import Any, ClassVar, Literal

from numpy import float64, interp, stack, zeros
from numpy.typing import NDArray
from quasar_typing.numpy import FloatMatrix, FloatVector
from scipy.sparse import csr_matrix


def _does_nothing(*args, **kwargs):
    pass


def _interp(
    x: NDArray[float64],
    template_x: NDArray[float64],
    _y: NDArray[float64],
) -> NDArray[float64]:
    return interp(x, template_x, _y, left=0.0, right=0.0)


def _interp2d(
    x: NDArray[float64],
    template_x: NDArray[float64],
    _derivs: NDArray[float64],
) -> NDArray[float64]:
    return stack([_interp(x, template_x, d) for d in _derivs], axis=0)


def _interp_matrix(
    _y: NDArray[float64],
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]],
) -> NDArray[float64]:
    M, b = interpolation_matrix
    return M @ _y + b


def _interp2d_matrix(
    _derivs: NDArray[float64],
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]],
) -> NDArray[float64]:
    M, b = interpolation_matrix
    return stack([M @ d + b for d in _derivs], axis=0)

###

class _TemplateModelBase:
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        simplify = self.simplify
        rescaling = self.rescaling
        return f"<{cls_name}({self.func_name}, {simplify=}, {rescaling=})>"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable:
        raise NotImplementedError

    def __init__(
        self,
        func_name: str | None = None,
        simplify: bool = False,
        rescaling: bool = False,
    ) -> None:
        self.func_name: str | None = func_name
        self.simplify: bool = simplify
        self.rescaling: bool = rescaling
        self.__wrapped__: Callable | None = self._get_wrapped(func_name)

    def __getstate__(self) -> dict:
        return {
            "func_name": self.func_name,
            "simplify": self.simplify,
            "rescaling": self.rescaling,
        }

    def __setstate__(self, state: dict) -> None:
        self.func_name = state["func_name"]
        self.simplify = state["simplify"]
        self.rescaling = state["rescaling"]
        self.__wrapped__ = self._get_wrapped(self.func_name)

class _TemplateEvaluate(_TemplateModelBase):
    def call_wrapped(
        self,
        template_x: FloatVector,
        args: tuple[Any,...],
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        if self.rescaling and y is not None:
            f = y
        else:
            f = zeros(template_x.size, dtype=float64)

        if self.__wrapped__ is not None:
            self.__wrapped__(f, *args)

        return f

    def __call__(
        self,
        x: FloatVector,
        template_x: FloatVector,
        args: tuple[Any,...],
        *,
        interpolation_matrix: tuple | Literal[False] | None = None,
        y: FloatVector | None = None,
    ) -> FloatVector:

        f = self.call_wrapped(
            template_x,
            args,
            y=y,
        )
        if self.rescaling:
            # Template is already evaluated on the wavelength grid -> no 
            # interpolation needed
            return f

        # Interpolate onto the wavelength grid
        if interpolation_matrix is None:
            f = _interp(x, template_x, f) 
        else:
            f = _interp_matrix(f, interpolation_matrix)

        # Add to existing flux density array, if possible
        if y is None:
            return f

        y += f
        return y

class _TemplateFitDeriv(_TemplateModelBase):
    _ndim: ClassVar[int]

    def call_wrapped(
        self,
        template_x,
        args: tuple[Any,...],
        *,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        if self.rescaling and derivs is not None:
            df = derivs
        else:
            df = zeros((self._ndim, template_x.size), dtype=float64)

        if self.__wrapped__ is not None:
            self.__wrapped__(df, *args)

        return df

    def __call__(
        self,
        x: FloatVector,
        template_x: FloatVector,
        args: tuple[Any,...],
        *,
        interpolation_matrix: tuple | Literal[False] | None = None,
        derivs: FloatMatrix | None = None,  
    ) -> FloatMatrix:
        df = self.call_wrapped(
            template_x,
            args,
            derivs=derivs,
        )
        if self.rescaling:
            return df

        # Interpolate onto the wavelength grid
        if interpolation_matrix is None:
            df = _interp2d(x, template_x, df) 
        else:
            df = _interp2d_matrix(df, interpolation_matrix)

        # Add to existing partial derivatives, if possible
        if derivs is None:
            return df

        derivs += df
        return derivs

        