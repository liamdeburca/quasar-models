from numpy import zeros_like, ones_like, float64, bool_, ones
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .utils import _split_fit_deriv, prepare_data_for_split
from ..utils.interpolation import get_template_transform
from ..utils.template import BaseTemplate

from quasar_utils.convolution import (
    kernel, kernel_deriv, convolve, convolve_signal, _identify_closest_idx, _identify_closest, SIGMA_TO_FWHM
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
    template: BaseTemplate = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> float | NDArray[float64]:
    """
    Evaluates the IronModel at the given x values.
    """
    _fwhm, _data = prepare_data_for_split(
        template,
        split, left, right, 
        sigma_res=sigma_res, scale=scale,
    )

    f = flux * convolve.__wrapped__(_data, _fwhm, fwhm, sigma_res)
    transform = get_template_transform(x, template.x, interpolation_matrix)

    return transform(f)

def evaluate_interp(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> float | NDArray[float64]:
    idx = _identify_closest_idx(template.fwhm, fwhm)

    if template.fwhm[idx] == fwhm:
        f = flux * template.data[idx]
    else:
        if idx == template.fwhm.size - 1:
            idx -= 1

        fwhm0, fwhm1 = template.fwhm[idx:idx+2]
        y0, y1 = flux * template.data[idx:idx+2]
        f = y0 + (y1 - y0) / (fwhm1 - fwhm0) * (fwhm - fwhm0)

    transform = get_template_transform(x, template.x, interpolation_matrix)
    
    return transform(f)

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
    template: BaseTemplate = None,
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

def evaluate_interp_sparse(
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
) -> tuple[NDArray[bool_], NDArray[float64]]:
    return (
        ones_like(x, dtype=bool_),
        evaluate_interp(
            x, 
            flux, fwhm, 
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
    template: BaseTemplate = None,
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

    if (fixed is not None) and not all(fixed.values()):
        transform = get_template_transform(x, template.x, interpolation_matrix)

        _fwhm, _data = prepare_data_for_split(
            template,
            split, left, right, 
            sigma_res=sigma_res, scale=scale,
        )
        fwhm_init, signal = _identify_closest(
            flux * _data, 
            _fwhm, 
            fwhm, 
            for_deriv=True,
        )
        if fwhm_init == fwhm:
            fwhm_kernel = 0 
            k = ones(1, dtype=float64)
            f = signal
        else:
            fwhm_kernel = (fwhm**2 - fwhm_init**2)**0.5
            k = kernel(fwhm_kernel, sigma_res)
            f = convolve_signal.__wrapped__(signal, k)

        if not fixed['flux'] and (flux > 0):
            df_dflux[:] = transform(f) / flux

        if not fixed['fwhm']:
            df_dfwhm[:] = transform(
                -SIGMA_TO_FWHM * signal \
                if fwhm_kernel == 0 else \
                convolve_signal.__wrapped__(
                    signal, 
                    kernel_deriv(fwhm_kernel, sigma_res),
                )
            )

        if not all(fixed[key] for key in ['split', 'left', 'right']):
            ds = _split_fit_deriv(
                template.x,
                split, left, right,
                sigma_res=sigma_res, scale=scale,
                fixed={key: fixed[key] for key in ['split', 'left', 'right']},
            )

            if not fixed['split']:
                df_dsplit[:] = transform(
                    ds[0] * signal \
                    if fwhm_kernel == 0 else \
                    convolve_signal.__wrapped__(ds[0] * signal, k)
                )

            if not fixed['left']:
                df_dleft[:] = transform(
                    ds[1] * signal \
                    if fwhm_kernel == 0 else \
                    convolve_signal.__wrapped__(ds[1] * signal, k)
                )

            if not fixed['right']:
                df_dright[:] = transform(
                    ds[2] * signal \
                    if fwhm_kernel == 0 else \
                     convolve_signal.__wrapped__(ds[2] * signal, k)
                )

    return [df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright]

def fit_deriv_interp(
    x: float | NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate = None,
    interpolation_matrix: tuple[csr_matrix, NDArray[float64]] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[NDArray[float64]]:
    df_dflux  = zeros_like(x, dtype=float64)
    df_dfwhm  = zeros_like(x, dtype=float64)

    # Note: other derivatives are zero by definition
    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    if (fixed is not None) and not all(fixed.values()):
        transform = get_template_transform(x, template.x, interpolation_matrix)

        idx = _identify_closest_idx(template.fwhm, fwhm, for_deriv=True)

        try:
            fwhm0, fwhm1 = template.fwhm[idx:idx+2]
            t0, t1 = template.data[idx:idx+2]
            grad = (t1 - t0) / (fwhm1 - fwhm0)
            t = t0 + grad * (fwhm - fwhm0)
        except:
            print(idx, template.fwhm.size, fwhm, template.fwhm[idx])
            raise ValueError

        if not fixed['flux']: 
            df_dflux[:] = transform(t)
        if not fixed['fwhm']: 
            df_dfwhm[:] = flux * transform(grad)

    return [df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright]