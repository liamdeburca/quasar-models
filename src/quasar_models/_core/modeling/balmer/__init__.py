__all__ = [
    "BalmerEvaluate",
    "BalmerFitDeriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
    "evaluate_exact",
    "evaluate_interp",
    "fit_deriv_exact_all",
    "fit_deriv_exact_flux_and_fwhm",
    "fit_deriv_exact_flux_and_ratio",
    "fit_deriv_exact_fwhm_and_ratio",
    "fit_deriv_exact_only_flux",
    "fit_deriv_exact_only_fwhm",
    "fit_deriv_exact_only_ratio",
    "fit_deriv_interp_all",
    "fit_deriv_interp_flux_and_fwhm",
    "fit_deriv_interp_flux_and_ratio",
    "fit_deriv_interp_fwhm_and_ratio",
    "fit_deriv_interp_only_flux",
    "fit_deriv_interp_only_fwhm",
    "fit_deriv_interp_only_ratio",
]
from .utils import (
    BalmerEvaluate,
    BalmerFitDeriv,
)


def choose_evaluate_func(allow_interp_fitting: bool) -> BalmerEvaluate:
    return evaluate_interp if allow_interp_fitting else evaluate_exact


def choose_fit_deriv_func(
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> BalmerFitDeriv:
    _case = (
        fixed.get("flux", False),
        fixed.get("fwhm", False),
        fixed.get("ratio", False),
    )
    if allow_interp_fitting:
        match _case:
            # General case
            case False, False, False:
                return fit_deriv_interp_all
            # Two free parameters
            case False, False, True:
                return fit_deriv_interp_flux_and_fwhm
            case False, True, False:
                return fit_deriv_interp_flux_and_ratio
            case True, False, False:
                return fit_deriv_interp_fwhm_and_ratio
            # One free parameter
            case False, True, True:
                return fit_deriv_interp_only_flux
            case True, False, True:
                return fit_deriv_interp_only_fwhm
            case True, True, False:
                return fit_deriv_interp_only_ratio

            case True, True, True:
                return does_nothing
    else:
        match _case:
            # General case
            case False, False, False:
                return fit_deriv_exact_all
            # Two free parameters
            case False, False, True:
                return fit_deriv_exact_flux_and_fwhm
            case False, True, False:
                return fit_deriv_exact_flux_and_ratio
            case True, False, False:
                return fit_deriv_exact_fwhm_and_ratio
            # One free parameter
            case False, True, True:
                return fit_deriv_exact_only_flux
            case True, False, True:
                return fit_deriv_exact_only_fwhm
            case True, True, False:
                return fit_deriv_exact_only_ratio

            case True, True, True:
                return does_nothing


### By CONVOLUTION

evaluate_exact = BalmerEvaluate("evaluate_exact")

fit_deriv_exact_only_flux = BalmerFitDeriv("fit_deriv_exact_only_flux")
fit_deriv_exact_only_fwhm = BalmerFitDeriv("fit_deriv_exact_only_fwhm")
fit_deriv_exact_only_ratio = BalmerFitDeriv("fit_deriv_exact_only_ratio")

fit_deriv_exact_flux_and_fwhm = BalmerFitDeriv("fit_deriv_exact_flux_and_fwhm")
fit_deriv_exact_flux_and_ratio = BalmerFitDeriv("fit_deriv_exact_flux_and_ratio")
fit_deriv_exact_fwhm_and_ratio = BalmerFitDeriv("fit_deriv_exact_fwhm_and_ratio")

fit_deriv_exact_all = BalmerFitDeriv("fit_deriv_exact_all")

### By INTERPOLATION

evaluate_interp = BalmerEvaluate("evaluate_interp")

fit_deriv_interp_only_flux = BalmerFitDeriv("fit_deriv_interp_only_flux")
fit_deriv_interp_only_fwhm = BalmerFitDeriv("fit_deriv_interp_only_fwhm")
fit_deriv_interp_only_ratio = BalmerFitDeriv("fit_deriv_interp_only_ratio")

fit_deriv_interp_flux_and_fwhm = BalmerFitDeriv("fit_deriv_interp_flux_and_fwhm")
fit_deriv_interp_flux_and_ratio = BalmerFitDeriv("fit_deriv_interp_flux_and_ratio")
fit_deriv_interp_fwhm_and_ratio = BalmerFitDeriv("fit_deriv_interp_fwhm_and_ratio")

fit_deriv_interp_all = BalmerFitDeriv("fit_deriv_interp_all")

does_nothing = BalmerFitDeriv()
