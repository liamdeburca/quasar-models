from numpy import interp, ndarray, zeros_like, ones_like, float64, bool_
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .utils import _split_evaluate, _split_fit_deriv
from ..utils.template import Template

from quasar_utils.convolution import (
    kernel, kernel_deriv, convolve, convolve_signal, _identify_closest_idx,
)

def evaluate(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float = None,
    scale: float = None,
    template: Template = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> float | NDArray[float64]:
    """
    Evaluates the IronModel at the given x values.
    """
    data = template.copy()

    if not (left == right == 1.0):
        data *= _split_evaluate(
            template.x, 
            split, left, right, 
            sigma_res=sigma_res, scale=scale,
        )[None,:]

    f = flux * convolve.__wrapped__(data, template.fwhm, fwhm, sigma_res)
    if not (isinstance(x, ndarray) and (x == template.x).all()):
        if interpolation_matrix is None:
            f = interp(x, template.x, f, left=0, right=0)
        else:
            M, b = interpolation_matrix
            f = M @ f + b

    return f

def evaluate_template(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: Template = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> float | NDArray[float64]:
    idx = _identify_closest_idx(template.fwhm, fwhm)

    if template.fwhm[idx] == fwhm:
        y = template.data[idx]
    else:
        fwhm0, fwhm1 = template.fwhm[idx:idx+2]
        y0, y1 = template.data[idx:idx+2]
        y = y0 + (y1 - y0) / (fwhm1 - fwhm0) * (fwhm - fwhm0)

    if isinstance(x, ndarray) and (x == template.x).all():
        f = flux * y
    elif isinstance(x, ndarray) and not (interpolation_matrix is None):
        M, b = interpolation_matrix
        f = flux * (M @ y + b)
    else:
        f = flux * interp(x, template.x, y, left=0, right=0)

    return f

def evaluate_sparse(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float = None,
    scale: float = None,
    template: Template = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> tuple[NDArray[bool_], NDArray[float64]]:
    return (
        ones_like(x, dtype=bool_),
        evaluate(
            x, 
            flux, fwhm, split, left, right, 
            sigma_res=sigma_res, scale=scale, 
            template=template, 
            interpolation_matrix=interpolation_matrix,
        )
    )

def fit_deriv(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float = None,
    scale: float = None,
    template: Template = None,
    fixed: dict[str, bool] | None = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> list[NDArray[float64]]:
    """
    Calculates the partial derivatives of the IronModel with respect to its 
    parameters: 'flux', 'fwhm', 'split', 'left', 'right'.
    """
    df_dflux  = zeros_like(x, dtype=float64)
    df_dfwhm  = zeros_like(x, dtype=float64)
    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    if (x == template.x).all():
        transform = lambda y: y
    elif interpolation_matrix is not None:
        M, b = interpolation_matrix
        transform = lambda y: M @ y + b
    else:
        transform = lambda y: interp(x, template.x, y, left=0, right=0)

    if not (fixed is None) and not all(fixed.values()):
        idx = _identify_closest_idx(template.fwhm, fwhm)
        fwhm_initial = template.fwhm[idx]
        signal = template.data[idx].copy()

        if not (left == right == 1.0):
            signal *= _split_evaluate(
                template.x,
                split, left, right,
                sigma_res=sigma_res, scale=scale,
            )

        fwhm_kernel = (fwhm**2 - fwhm_initial**2)**0.5
        k = kernel(fwhm_kernel, sigma_res)
        _f = convolve_signal.__wrapped__(signal, k)

        if not fixed['flux']:
            df_dflux[:] = transform(_f)

        if not fixed['fwhm']:
            dk: ndarray[float] = kernel_deriv(fwhm_kernel, sigma_res)
            df_dfwhm[:] = flux \
                * transform(convolve_signal.__wrapped__(signal, dk))

        if not all(fixed[key] for key in ['split', 'left', 'right']):
            ds: list[ndarray[float]] = _split_fit_deriv(
                template.x,
                split, left, right,
                sigma_res=sigma_res, scale=scale,
                fixed={key: fixed[key] for key in ['split', 'left', 'right']},
            )

            if not fixed['split']:
                df_dsplit[:] = flux \
                    * transform(convolve_signal.__wrapped__(ds[0] * signal, k))

            if not fixed['left']:
                df_dleft[:] = flux \
                    * transform(convolve_signal.__wrapped__(ds[1] * signal, k))

            if not fixed['right']:
                df_dright[:] = flux \
                    * transform(convolve_signal.__wrapped__(ds[2] * signal, k))

    return [df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright]

def fit_deriv_template(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: Template = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    df_dflux  = zeros_like(x, dtype=float64)
    df_dfwhm  = zeros_like(x, dtype=float64)

    # Note: other derivatives are zero by definition
    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    if not (fixed is None) and not all(fixed.values()):
        idx = _identify_closest_idx(template.fwhm, fwhm)

        fwhm0, fwhm1 = template.fwhm[idx:idx+2]
        y0, y1 = flux * template.data[idx:idx+2]
        grad = (y1 - y0) / (fwhm1 - fwhm0)
        y = y0 + grad * (fwhm - fwhm0)

        if not fixed['flux']:
            if (x == template.x).all():
                df_dflux[:] = y / flux
            elif interpolation_matrix is not None:
                M, b = interpolation_matrix
                df_dflux[:] = (M @ y + b) / flux
            else:
                df_dflux[:] = interp(x, template.x, y, left=0, right=0) / flux

        if not fixed['fwhm']:
            df_dfwhm[:] = grad

    return [df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright]