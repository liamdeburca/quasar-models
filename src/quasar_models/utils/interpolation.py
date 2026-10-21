from collections.abc import Callable

from numpy import array_equal, interp
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import CSRMatrix


def get_template_transform(
    x: FloatVector,
    template_x: FloatVector,
    interpolation_matrix: tuple[CSRMatrix, FloatVector] | None,
) -> Callable[[FloatVector], FloatVector]:
    if array_equal(x, template_x):
        return lambda y: y
    elif interpolation_matrix is None:
        return lambda y: interp(x, template_x, y, left=0, right=0)
    else:
        M, b = interpolation_matrix
        return lambda y: M @ y + b
