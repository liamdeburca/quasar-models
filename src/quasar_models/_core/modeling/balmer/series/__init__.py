from numpy import array, float64, isclose, zeros
from numpy.typing import NDArray

from .evaluate import evaluate as _evaluate


def evaluate(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    sigma_res: float,
    waves: NDArray[float64],
    weights: NDArray[float64],
    edge: float | None = None,
    normalisation: float | None = None,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    assert isclose(weights.sum(), 1.0)

    if normalisation is None:
        assert edge is not None
        _y = zeros(1, dtype=float64)
        _x = array([edge], dtype=float64)
        _evaluate(_y, _x, 1.0, fwhm, sigma_res, waves, weights)
        normalisation = _y[0]
        del _y, _x

    if y is None:
        y = zeros(x.shape, dtype=float64)
    _evaluate(
        y,
        x,
        flux / normalisation,
        fwhm,
        sigma_res,
        waves,
        weights,
    )
    return y
