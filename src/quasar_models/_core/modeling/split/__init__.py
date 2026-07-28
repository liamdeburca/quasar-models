__all__ = [
    "evaluate",
    "fit_deriv_all",
    "fit_deriv_left_and_right",
    "fit_deriv_only_left",
    "fit_deriv_only_right",
    "fit_deriv_only_split",
    "fit_deriv_split_and_left",
    "fit_deriv_split_and_right",
]

from numpy import float64, zeros
from numpy.typing import NDArray

from .evaluate import evaluate as _evaluate
from .fit_deriv import (
    fit_deriv_all as _fit_deriv_all,
)
from .fit_deriv import (
    fit_deriv_left_and_right as _fit_deriv_left_and_right,
)
from .fit_deriv import (
    fit_deriv_only_left as _fit_deriv_only_left,
)
from .fit_deriv import (
    fit_deriv_only_right as _fit_deriv_only_right,
)
from .fit_deriv import (
    fit_deriv_only_split as _fit_deriv_only_split,
)
from .fit_deriv import (
    fit_deriv_split_and_left as _fit_deriv_split_and_left,
)
from .fit_deriv import (
    fit_deriv_split_and_right as _fit_deriv_split_and_right,
)


def evaluate(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if y is None:
        y = zeros(x.size, dtype=float64)
    _evaluate(y, x, split, left, right, sigma_res, scale)
    return y


def fit_deriv_only_split(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_only_split(derivs, x, split, left, right, sigma_res, scale)
    return derivs


def fit_deriv_only_left(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_only_left(derivs, x, split, left, right, sigma_res, scale)
    return derivs


def fit_deriv_only_right(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_only_right(derivs, x, split, left, right, sigma_res, scale)
    return derivs


def fit_deriv_split_and_left(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_split_and_left(derivs, x, split, left, right, sigma_res, scale)
    return derivs


def fit_deriv_split_and_right(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_split_and_right(derivs, x, split, left, right, sigma_res, scale)
    return derivs


def fit_deriv_left_and_right(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_left_and_right(derivs, x, split, left, right, sigma_res, scale)
    return derivs


def fit_deriv_all(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if derivs is None:
        derivs = zeros((3, x.size), dtype=float64)
    _fit_deriv_all(derivs, x, split, left, right, sigma_res, scale)
    return derivs
