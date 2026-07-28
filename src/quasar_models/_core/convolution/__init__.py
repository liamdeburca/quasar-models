__all__ = [
    "convolve",
    "convolve_deriv",
    "convolve_signal",
    "convolve_signal2d",
    "identify_closest_idx",
    "identify_closest_idx_for_deriv",
    "kernel",
    "kernel_deriv",
    "pixels",
    "scale",
]

from numpy import float64
from numpy.typing import NDArray

from .utils import (
    convolve as _convolve,
)
from .utils import (
    convolve_deriv as _convolve_deriv,
)
from .utils import (
    convolve_signal as _convolve_signal,
)
from .utils import (
    convolve_signal2d as _convolve_signal2d,
)
from .utils import (
    identify_closest_idx as _identify_closest_idx,
)
from .utils import (
    identify_closest_idx_for_deriv as _identify_closest_idx_for_deriv,
)
from .utils import (
    kernel as _kernel,
)
from .utils import (
    kernel_deriv as _kernel_deriv,
)
from .utils import (
    pixels as _pixels,
)
from .utils import (
    scale as _scale,
)


def scale(
    fwhm: float,
    sigma_res: float,
) -> float:
    return _scale(fwhm, sigma_res)


def pixels(
    scale: float,
    n_scales: float,
) -> NDArray[float64]:
    return _pixels(scale, n_scales)


def kernel(
    fwhm: float,
    sigma_res: float,
    n_scales: float,
) -> NDArray[float64]:
    return _kernel(fwhm, sigma_res, n_scales)


def kernel_deriv(
    fwhm: float,
    sigma_res: float,
    n_scales: float,
) -> NDArray[float64]:
    return _kernel_deriv(fwhm, sigma_res, n_scales)


def convolve_signal(
    signal: NDArray[float64],
    kernel: NDArray[float64],
) -> NDArray[float64]:
    return _convolve_signal(signal, kernel)


def convolve_signal2d(
    signal: NDArray[float64],
    kernel: NDArray[float64],
) -> NDArray[float64]:
    return _convolve_signal2d(signal, kernel)


def identify_closest_idx(
    fwhm: NDArray[float64],
    fwhm_final: float,
) -> int:
    return _identify_closest_idx(fwhm, fwhm_final)


def identify_closest_idx_for_deriv(
    fwhm: NDArray[float64],
    fwhm_final: float,
) -> int:
    return _identify_closest_idx_for_deriv(fwhm, fwhm_final)


def convolve(
    data: NDArray[float64],
    fwhm: NDArray[float64],
    fwhm_final: float,
    sigma_res: float,
    n_scales: float,
) -> NDArray[float64]:
    return _convolve(data, fwhm, fwhm_final, sigma_res, n_scales)


def convolve_deriv(
    data: NDArray[float64],
    fwhm: NDArray[float64],
    fwhm_final: float,
    sigma_res: float,
    n_scales: float,
) -> NDArray[float64]:
    return _convolve_deriv(data, fwhm, fwhm_final, sigma_res, n_scales)
