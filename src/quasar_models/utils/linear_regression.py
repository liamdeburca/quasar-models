from pydantic import validate_call
from numpy import full_like, nan, einsum, float64

from quasar_typing.numpy import FloatVector, FittableFloatVector, \
    FittableFloatMatrix

@validate_call(validate_return=False)
def linreg(
    x: FittableFloatVector | FittableFloatMatrix,
    y: FittableFloatVector | FittableFloatMatrix,
    dy: FittableFloatVector | FittableFloatMatrix,
) -> tuple[float | FloatVector]:
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
        return y.mean(), 0
    
    return (C * ya - B * yb) / det, (A * yb - B * ya) / det