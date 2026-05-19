from logging import getLogger
from numpy import full_like, nan, einsum, zeros_like, float64

from quasar_utils.decorators import validate_call
from quasar_typing.numpy import FloatVector, FittableFloatVector, \
    FittableFloatMatrix

logger = getLogger(__name__)

@validate_call
def linreg(
    x: FittableFloatVector | FittableFloatMatrix,
    y: FittableFloatVector | FittableFloatMatrix,
    dy: FittableFloatVector | FittableFloatMatrix,
) -> tuple[float, float] | tuple[FloatVector, FloatVector]:
    """
    Performs weighted least-squares regression efficiently. 

    Parameters
    ----------
    x : numpy.array
        Array of coordinates along the first axis. 
    y : numpy.array
        Array of coordinates along the second axis. 
    dy : numpy.array
        Array of coordinate uncertainties along the second axis. 

    Returns
    -------
    intercept : float
        The 0th polynomial coefficient, equivalent to the y-intercept of the 
        fitted line. 
    gradient : float
        The 1st polynomial coefficient, equivalent to the gradient of the 
        fitted line. 
    """
    assert x.shape == y.shape == dy.shape
    
    if x.shape[-1] == 0:
        msg = "Cannot perform linear regression on empty arrays."
        logger.critical(msg)
        raise ValueError(msg)
    elif x.shape[-1] == 1:
        msg = "Cannot perform linear regression on single-element arrays. " \
            "Defaulting to: intercept=y[0], gradient=0."
        logger.warning(msg)
        return y[0], zeros_like(y[0], dtype=float64)

    a = (1 / dy)**2
    b = x * a
    c = x * b

    A = einsum("...i -> ...", a)
    B = einsum("...i -> ...", b)
    C = einsum("...i -> ...", c)

    det = A * C - B**2
    ya = einsum("...i,...i -> ...", y, a)
    yb = einsum("...i,...i -> ...", y, b)

    if x.ndim != 1:
        intercept = full_like(det, fill_value=nan, dtype=float64)
        gradient  = full_like(det, fill_value=nan, dtype=float64)

        mask = (det != 0)
        intercept[mask] = (C[mask] * ya[mask] - B[mask] * yb[mask]) / det[mask]
        gradient[mask]  = (A[mask] * yb[mask] - B[mask] * ya[mask]) / det[mask]

        return intercept, gradient
        
    if det == 0:
        msg = "Determinant is zero. Defaulting to: " \
            "intercept=y.mean(), gradient=0."
        logger.warning(msg)
        return y.mean(), 0
    
    return (C * ya - B * yb) / det, (A * yb - B * ya) / det