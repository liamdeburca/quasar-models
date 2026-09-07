__all__ = [
    "HostGalaxyEvaluate",
    "HostGalaxyFitDeriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
    "evaluate_exact",
    "evaluate_interp",
    "fit_deriv_exact_all",
    "fit_deriv_exact_only_flux",
    "fit_deriv_exact_only_fwhm",
    "fit_deriv_interp_all",
    "fit_deriv_interp_only_flux",
    "fit_deriv_interp_only_fwhm",
]
from .utils import (
    HostGalaxyEvaluate,
    HostGalaxyFitDeriv,
)


def choose_evaluate_func(allow_interp_fitting: bool) -> HostGalaxyEvaluate:
    return evaluate_interp if allow_interp_fitting else evaluate_exact


def choose_fit_deriv_func(
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> HostGalaxyFitDeriv:

    match allow_interp_fitting, fixed.get("flux", False), fixed.get("fwhm", False):
        case True, False, False:
            return fit_deriv_interp_all
        case True, False, True:
            return fit_deriv_interp_only_flux
        case True, True, False:
            return fit_deriv_interp_only_fwhm

        case False, False, False:
            return fit_deriv_exact_all
        case False, False, True:
            return fit_deriv_exact_only_flux
        case False, True, False:
            return fit_deriv_exact_only_fwhm

        case _, True, True:
            return does_nothing


### By CONVOLUTION: Simplify template

evaluate_exact = HostGalaxyEvaluate(
    "evaluate_exact",
    simplify=True,
)

fit_deriv_exact_only_flux = HostGalaxyFitDeriv(
    "fit_deriv_exact_only_flux", 
    simplify=True,
)
fit_deriv_exact_only_fwhm = HostGalaxyFitDeriv(
    "fit_deriv_exact_only_fwhm",
    simplify=True,
)
fit_deriv_exact_all = HostGalaxyFitDeriv(
    "fit_deriv_exact_all",
    simplify=True,
)

### By INTERPOLATION: No simplification

evaluate_interp = HostGalaxyEvaluate(
    "evaluate_interp",
    simplify=False,
)

fit_deriv_interp_only_flux = HostGalaxyFitDeriv(
    "fit_deriv_interp_only_flux",
    simplify=False,
)
fit_deriv_interp_only_fwhm = HostGalaxyFitDeriv(
    "fit_deriv_interp_only_fwhm",
    simplify=False,
)
fit_deriv_interp_all = HostGalaxyFitDeriv(
    "fit_deriv_interp_all",
    simplify=False,
)

does_nothing = HostGalaxyFitDeriv()
