from numpy import clip, log, exp, zeros_like, float64, arange
from numpy.typing import NDArray

from ..utils.template import BaseTemplate

def _split_evaluate(
    x: float | NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float = None,
    scale: float = None,
) -> float | NDArray[float64]:
    """
    Calculates the split weight at the given x values.
    """  
    assert sigma_res is not None, "sigma_res must be provided"
    assert scale is not None, "scale must be provided"
    
    z = clip(log(split / x) / (scale * sigma_res), -5, 5)
    s = 1 / (1 + exp(z))

    return (right - left) * s + left

def _split_fit_deriv(
    x: NDArray[float64],
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float = None,
    scale: float = None,
    fixed: dict[str, bool] = None,
) -> list[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Calculates the partial derivatives of the split function with respect to the
    parameters: 'split', 'left', 'right'.
    """
    assert sigma_res is not None, "sigma_res must be provided"
    assert scale is not None, "scale must be provided"

    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    if fixed is not None and not all(fixed.values()):
        _exp_z = exp(clip(log(split / x) / (scale * sigma_res), -5, 5))
        s = 1 / (1 + _exp_z)

        if not fixed['split']:
            ds_dz = -s**2 * _exp_z
            dz_dsplit = 1 / (scale * sigma_res * split)
            df_dsplit[:] = (right - left) * ds_dz * dz_dsplit

        if not fixed['left']:
            df_dleft[:] = -s + 1

        if not fixed['right']:
            df_dright[:] = s

    return [df_dsplit, df_dleft, df_dright]

def prepare_data_for_split(
    template: BaseTemplate,
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float = None,
    scale: float = None,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Convenience function preparing the fwhm and data arrays of a template for
    cases where applying a split is relevant. 

    If a split is relevant: 
    - The template's original fwhm and data arrays are returned

    If a split is applied:
    - The template's original fwhm and data arrays are cropped s.t. only the
      smallest FWHM and the corresponding data array are saved. 
    - The cropped data array is then multiplied by a split weight function. 

    This approach ensures that convolved signals are as correct as possible. 
    """
    if left == right == 1.0: return template.fwhm, template.data
    
    _fwhm = template.fwhm[:1]
    _data = template.data[:1] * _split_evaluate(
        template.x,
        split, left, right,
        scale=scale, sigma_res=sigma_res,
    )
    return _fwhm, _data

# For loading/parsing Fe templates

def _get_xlog(
    x_bounds: tuple[float, float],
    sigma_res: float,
) -> NDArray[float64]:
    x0, x1 = x_bounds
    n = log(x1 / x0) // log(1 + sigma_res) + 1
    return x0 * (1 + sigma_res)**arange(n)