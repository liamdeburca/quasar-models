from numpy import allclose, array, diff, float64, zeros
from numpy.typing import NDArray

from .evaluate import (
    attenuation as _attenuation,
)
from .evaluate import (
    continuum as _continuum,
)
from .evaluate import (
    evaluate as _evaluate,
)


def attenuation(
    x: NDArray[float64],
    *,
    tau: float,
    scale: float,
    edge: float,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if y is None:
        y = zeros(x.shape, dtype=float64)
    _attenuation(y, x, tau, scale, edge)
    return y


def continuum(
    x: NDArray[float64],
    flux: float,
    *,
    temp: float,
    edge: float,
    boltz: float,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if y is None:
        y = zeros(x.shape, dtype=float64)
    _continuum(y, x, flux, temp, edge, boltz)
    return y


def evaluate(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    temp: float,
    tau: float,
    scale: float,
    edge: float,
    boltz: float,
    sigma_res: float,
    n_scales: float,
    normalisation: float | None = None,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if fwhm > 0.0:
        assert sigma_res is not None
        assert n_scales is not None
        assert allclose(diff(x) / x[:-1], sigma_res)

    if normalisation is None:
        _y = zeros(1, dtype=float64)
        _x = array([edge], dtype=float64)
        _evaluate(_y, _x, 1.0, fwhm, temp, tau, scale, edge, boltz, sigma_res, n_scales)
        normalisation = _y[0]
        del _y, _x

    if y is None:
        y = zeros(x.shape, dtype=float64)
    _evaluate(
        y,
        x,
        flux / normalisation,
        fwhm,
        temp,
        tau,
        scale,
        edge,
        boltz,
        sigma_res,
        n_scales,
    )
    return y
