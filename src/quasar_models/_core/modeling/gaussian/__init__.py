__all__ = [
    "GaussianEvaluate", "GaussianFitDeriv",
    "choose_evaluate_func", "choose_fit_deriv_func",

    "evaluate_v", "prime_v",
    
    "fit_deriv_v_only_strength",
    "fit_deriv_v_only_fwhm_v",
    "fit_deriv_v_only_v_off",
    "fit_deriv_v_strength_and_fwhm_v",
    "fit_deriv_v_strength_and_v_off",
    "fit_deriv_v_fwhm_v_and_v_off",
    "fit_deriv_v_all",
]
from .utils import (
    GaussianEvaluate, GaussianFitDeriv,
)

def choose_evaluate_func() -> GaussianEvaluate:
    return evaluate_v

def choose_fit_deriv_func(fixed: dict[str, bool]) -> GaussianFitDeriv:
    match fixed.get('strength', False), fixed.get('fwhm_v', False), fixed.get('v_off', False):
        case False, False, False:
            return fit_deriv_v_all
        
        case False, True, True:
            return fit_deriv_v_only_strength
        case True, False, True:
            return fit_deriv_v_only_fwhm_v
        case True, True, False:
            return fit_deriv_v_only_v_off
        
        case False, False, True:
            return fit_deriv_v_strength_and_fwhm_v
        case False, True, False:
            return fit_deriv_v_strength_and_v_off
        case True, False, False:
            return fit_deriv_v_fwhm_v_and_v_off
        
        case _:
            raise ValueError(f"{fixed=}")

###

evaluate_v = GaussianEvaluate('evaluate_v')

fit_deriv_v_only_strength = GaussianFitDeriv('fit_deriv_v_only_strength')
fit_deriv_v_only_fwhm_v = GaussianFitDeriv('fit_deriv_v_only_fwhm_v')
fit_deriv_v_only_v_off = GaussianFitDeriv('fit_deriv_v_only_v_off')

fit_deriv_v_strength_and_fwhm_v = GaussianFitDeriv('fit_deriv_v_strength_and_fwhm_v')
fit_deriv_v_strength_and_v_off = GaussianFitDeriv('fit_deriv_v_strength_and_v_off')
fit_deriv_v_fwhm_v_and_v_off = GaussianFitDeriv('fit_deriv_v_fwhm_v_and_v_off')

fit_deriv_v_all = GaussianFitDeriv('fit_deriv_v_all')

###

prime_v = GaussianEvaluate('prime_v')