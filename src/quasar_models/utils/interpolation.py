from typing import Callable
from numpy import array_equal, interp, float64
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

def get_template_transform(
    x: NDArray[float64],
    template_x: NDArray[float64],
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None,
) -> Callable[[NDArray[float64]], NDArray[float64]]:
    if array_equal(x, template_x):
        return lambda y: y
    elif interpolation_matrix is None:
        return lambda y: interp(x, template_x, y, left=0, right=0)
    else:
        M, b = interpolation_matrix
        return lambda y: M @ y + b