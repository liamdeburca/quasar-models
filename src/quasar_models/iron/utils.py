from numpy import clip, log, exp, zeros_like, float64
from numpy.typing import NDArray

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