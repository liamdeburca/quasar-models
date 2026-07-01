__all__ = [
    "evaluate",
    "fit_deriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
]

from typing import Protocol
from numpy import zeros_like, float64, ones, clip, log, exp

from .utils import _split_fit_deriv
from ..utils.interpolation import get_template_transform
from ..utils.template import BaseTemplate, evaluation as template_evaluation

from quasar_utils.convolution import (
    kernel, kernel_deriv, convolve_signal, _identify_closest, SIGMA_TO_FWHM
)

from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

class EvaluateFunc(Protocol):
    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        template: BaseTemplate,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> FloatVector:
        ...

class FitDerivFunc(Protocol):
    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        template: BaseTemplate,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        fixed: dict[str, bool] | None = None,     
    ) -> list[FloatVector]:
        ...

### SPLIT ###

def split_evaluate(
    x: FloatVector,
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
) -> FloatVector:
    z = clip(log(split / x) / (scale * sigma_res), -5, 5)
    s = 1 / (1 + exp(z))
    return (right - left) * s + left

def split_fit_deriv(
    x: FloatVector,
    split: float,
    left: float,
    right: float,
    *,
    sigma_res: float,
    scale: float,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    if fixed is None:
        fixed = {'split': False, 'left': False, 'right': False}

    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    if not all(fixed.values()):
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

### By CONVOLUTION

def evaluate_exact(
    x: FloatVector,
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    template: BaseTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    if flux == 0:
        return zeros_like(x, dtype=float64)
    
    kwargs = {'interpolation_matrix': interpolation_matrix}
    if not ((split < template.x[0]) or (template.x[-1] < split) or (left == right)):
        kwargs['template_fwhm'] = template.fwhm[:1]
        kwargs['template_data'] = template.data[:1] * split_evaluate(
            template.x, split, left, right, 
            sigma_res=template.sigma_res, 
            scale=template.scale,
        )[None,:]

    return template_evaluation.evaluate(
        x, flux, fwhm,
        template=template,
        **kwargs,
    )

def fit_deriv_exact(
    x: FloatVector,
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    template: BaseTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    """
    Calculates the partial derivatives of the IronModel with respect to its 
    parameters: 'flux', 'fwhm', 'split', 'left', 'right'.
    """
    if fixed is None:
        fixed = {
            'flux': False,
            'fwhm': False,
            'split': False,
            'left': False,
            'right': False,
        }
    df_dflux = zeros_like(x, dtype=float64)
    df_dfwhm = zeros_like(x, dtype=float64)
    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    if not all(fixed.values()):
        transform = get_template_transform(x, template.x, interpolation_matrix)

        if (split < template.x[0]) or (template.x[-1] < split) or (left == right):
            _fwhm = template.fwhm
            _data = template.data
        else:
            _fwhm = template.fwhm[:1]
            _data = template.data[:1] * split_evaluate(
                template.x, split, left, right, 
                sigma_res=template.sigma_res, 
                scale=template.scale,
            )[None,:]

        fwhm_init, signal = _identify_closest(
            _data, _fwhm, fwhm, 
            for_deriv=True,
        )
        if fwhm_init == fwhm:
            fwhm_kernel = 0 
            k = ones(1, dtype=float64)
            f = signal
        else:
            fwhm_kernel = (fwhm**2 - fwhm_init**2)**0.5
            k = kernel(fwhm_kernel, template.sigma_res)
            f = convolve_signal.__wrapped__(signal, k)

        inv_norm = 1 / template.normalisation

        if not fixed['flux']:
            df_dflux[:] = transform(f) * inv_norm

        if not fixed['fwhm'] and flux != 0:
            df_dfwhm[:] = flux * inv_norm * transform(
                -SIGMA_TO_FWHM * signal \
                if fwhm_kernel == 0 else \
                convolve_signal.__wrapped__(
                    signal, 
                    kernel_deriv(fwhm_kernel, template.sigma_res),
                )
            )

        if not all(fixed[key] for key in ['split', 'left', 'right']) \
            and flux != 0 \
            and not ((split < template.x[0]) or (template.x[-1] < split) or (left == right)):
            ds = _split_fit_deriv(
                template.x,
                split, left, right,
                sigma_res=template.sigma_res, 
                scale=template.scale,
                fixed={key: fixed[key] for key in ['split', 'left', 'right']},
            )

            if not fixed['split'] and flux != 0:
                df_dsplit[:] = flux * inv_norm * transform(
                    ds[0] * signal \
                    if fwhm_kernel == 0 else \
                    convolve_signal.__wrapped__(ds[0] * signal, k)
                )

            if not fixed['left'] and flux != 0:
                df_dleft[:] = flux * inv_norm * transform(
                    ds[1] * signal \
                    if fwhm_kernel == 0 else \
                    convolve_signal.__wrapped__(ds[1] * signal, k)
                )

            if not fixed['right'] and flux != 0:
                df_dright[:] = flux * inv_norm * transform(
                    ds[2] * signal \
                    if fwhm_kernel == 0 else \
                     convolve_signal.__wrapped__(ds[2] * signal, k)
                )

    return [df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright]

### By INTERPOLATION

def evaluate_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    *,
    template: BaseTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
) -> FloatVector:
    
    if flux == 0:
        return zeros_like(x, dtype=float64)
    
    return template_evaluation.evaluate_interp(
        x, flux, fwhm,
        template=template,
        interpolation_matrix=interpolation_matrix,
    )

def fit_deriv_interp(
    x: FloatVector,
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    template: BaseTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    if fixed is None:
        fixed = {
            'flux': False,
            'fwhm': False,
            'split': False,
            'left': False,
            'right': False,
        }

    # Note: other derivatives are zero by definition
    df_dsplit = zeros_like(x, dtype=float64)
    df_dleft  = zeros_like(x, dtype=float64)
    df_dright = zeros_like(x, dtype=float64)

    df_dflux, df_dfwhm = template_evaluation.fit_deriv_interp(
        x, flux, fwhm,
        template=template,
        interpolation_matrix=interpolation_matrix,
        fixed={p: fixed[p] for p in ['flux', 'fwhm']},
    )
    return [df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright]

###

def choose_evaluate_func(
    perform_interp_fitting: bool,
) -> EvaluateFunc:
    return evaluate_interp if perform_interp_fitting else evaluate_exact

def evaluate(
    x: FloatVector,
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    template: BaseTemplate,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    perform_interp_fitting: bool | None = None,
    evaluate_func: EvaluateFunc | None = None,
) -> FloatVector:
    if evaluate_func is None:
        assert perform_interp_fitting is not None
        evaluate_func = choose_evaluate_func(perform_interp_fitting)

    return evaluate_func(
        x, flux, fwhm, split, left, right,
        template=template,
        interpolation_matrix=interpolation_matrix,
    )

def choose_fit_deriv_func(
    fixed: dict[str, bool] | None,
    perform_interp_fitting: bool,
) -> FitDerivFunc:
    if fixed is None:
        fixed = {
            'flux': False,
            'fwhm': False,
            'split': False,
            'left': False,
            'right': False,
        }

    return fit_deriv_interp if perform_interp_fitting else fit_deriv_exact

def fit_deriv(
    x: FloatVector,
    flux: float,
    fwhm: float,
    split: float,
    left: float,
    right: float,
    *,
    template: BaseTemplate,
    perform_interp_fitting: bool | None = None,
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    fixed: dict[str, bool] | None = None,
    fit_deriv_func: FitDerivFunc | None = None,
) -> list[FloatVector]:
    if fit_deriv_func is None:
        assert perform_interp_fitting is not None
        fit_deriv_func = choose_fit_deriv_func(fixed, perform_interp_fitting)

    return fit_deriv_func(
        x,
        flux, fwhm, split, left, right,
        template=template,
        interpolation_matrix=interpolation_matrix,
        fixed=fixed,
    )