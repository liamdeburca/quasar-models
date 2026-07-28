__all__ = [
    "_does_nothing",
    "_interp",
    "_interp2d",
    "_interp2d_matrix",
    "_interp_matrix",
]

from numpy import float64, interp, stack
from numpy.typing import NDArray
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
