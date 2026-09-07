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

from quasar_typing.numpy import FloatMatrix, FloatVector

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


def scale(fwhm: float, sigma_res: float) -> float:
    """
    Parameters
    ----------
    fwhm : float
        FWHM (km/s) of the template.
    sigma_res : float
        Velocity resolution (c) of the spectrum.
    
    Returns
    -------
    float
    """
    return _scale(fwhm, sigma_res)


def pixels(scale: float, n_scales: float) -> FloatVector:
    """
    Parameters
    ----------
    scale : float
    n_scales : float

    Returns
    -------
    FloatVector
    """
    return _pixels(scale, n_scales)


def kernel(fwhm: float, sigma_res: float, n_scales: float) -> FloatVector:
    """
    Calculate the Gaussian kernel used for convolution.

    Parameters
    ----------
    fwhm : float
        FWHM (km/s) of the template.
    sigma_res : float
        Velocity resolution (c) of the spectrum.
    n_scales : float

    Returns
    -------
    FloatVector
    """
    return _kernel(fwhm, sigma_res, n_scales)


def kernel_deriv(fwhm: float, sigma_res: float, n_scales: float) -> FloatVector:
    """
    Calculate the derivative of the Gaussian kernel used for convolution w.r.t. 
    FWHM.

    Parameters
    ----------
    fwhm : float
        FWHM (km/s) of the template.
    sigma_res : float
        Velocity resolution (c) of the spectrum.
    n_scales : float

    Returns
    -------
    FloatVector
    """
    return _kernel_deriv(fwhm, sigma_res, n_scales)


def convolve_signal(signal: FloatVector, kernel: FloatVector) -> FloatVector:
    """
    Convolve a signal with a kernel.

    Parameters
    ----------
    signal : FloatVector
        Signal to be convolved.
    kernel : FloatVector
        Kernel to convolve with the signal.

    Returns
    -------
    FloatVector
        Convolved signal.
    """
    return _convolve_signal(signal, kernel)


def convolve_signal2d(signal2d: FloatMatrix, kernel: FloatVector) -> FloatVector:
    """
    Convolve a 2D signal with a kernel.

    Parameters
    ----------
    signal2d : FloatMatrix
        2D signal to be convolved.
    kernel : FloatVector
        Kernel to convolve with the signal.

    Returns
    -------
    FloatVector
        Convolved signal.
    """
    return _convolve_signal2d(signal2d, kernel)


def identify_closest_idx(fwhm: FloatVector, fwhm_final: float) -> int:
    """
    Find the index of the closest FWHM value in an array to a given FWHM value.

    Parameters
    ----------
    fwhm : FloatVector
        Array of FWHM values.
    fwhm_final : float
        Target FWHM value to find the closest index for.

    Returns
    -------
    int
        Index of the closest element.
    """
    return _identify_closest_idx(fwhm, fwhm_final)


def identify_closest_idx_for_deriv(fwhm: FloatVector, fwhm_final: float) -> int:
    """
    Find the index of the closest FWHM value in an array to a given FWHM value.

    This function is specifically used for derivative calculations as it ensures 
    that if 'fwhm_final' lies in the 'fwhm' array, the index is shifted 
    downwards to guarantee convolution.

    Parameters
    ----------
    fwhm : FloatVector
        Array of FWHM values.
    fwhm_final : float
        Target FWHM value to find the closest index for.

    Returns
    -------
    int
        Index of the element.
    """
    return _identify_closest_idx_for_deriv(fwhm, fwhm_final)


def convolve(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
    n_scales: float,
) -> FloatVector:
    """
    Find the index of the closest FWHM value and convolve the corresponding data 
    array.

    Parameters
    ----------
    data : FloatMatrix
        2D array of data to be convolved.
    fwhm : FloatVector
        Array of FWHM values corresponding to the data.
    fwhm_final : float
        Target FWHM value for convolution.
    sigma_res : float
        Velocity resolution (c) of the spectrum.
    n_scales : float

    Returns
    -------
    FloatVector
        Convolved data array.
    """
    return _convolve(data, fwhm, fwhm_final, sigma_res, n_scales)


def convolve_deriv(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
    n_scales: float,
) -> FloatVector:
    """
    Find the index the closest FWHM value and calculate the derivative of the 
    convolved data array w.r.t. FWHM.

    Parameters
    ----------
    data : FloatMatrix
        2D array of data to be convolved.
    fwhm : FloatVector
        Array of FWHM values corresponding to the data.
    fwhm_final : float
        Target FWHM value for convolution.
    sigma_res : float
        Velocity resolution (c) of the spectrum.
    n_scales : float

    Returns
    -------
    FloatVector
        Derivative of the convolved data array w.r.t. FWHM.
    """
    return _convolve_deriv(data, fwhm, fwhm_final, sigma_res, n_scales)
