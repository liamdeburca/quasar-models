__all__ = [
    "PowerLawEvaluate",
    "PowerLawFitDeriv",
    "PowerLawInverse",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
    "does_nothing",
    "evaluate",
    "fit_deriv_all",
    "fit_deriv_only_alpha",
    "fit_deriv_only_flux",
    "inverse",
]

from .utils import (
    PowerLawEvaluate,
    PowerLawFitDeriv,
    PowerLawInverse,
)


def choose_evaluate_func() -> PowerLawEvaluate:
    return evaluate


def choose_fit_deriv_func(fixed: dict[str, bool]) -> PowerLawFitDeriv:
    match fixed.get("flux", False), fixed.get("alpha", False):
        case False, False:
            return fit_deriv_all
        case False, True:
            return fit_deriv_only_flux
        case True, False:
            return fit_deriv_only_alpha
        case _:
            return does_nothing


###

evaluate = PowerLawEvaluate("evaluate")
inverse = PowerLawInverse("inverse")

fit_deriv_only_flux = PowerLawFitDeriv("fit_deriv_only_flux")
fit_deriv_only_alpha = PowerLawFitDeriv("fit_deriv_only_alpha")
fit_deriv_all = PowerLawFitDeriv("fit_deriv_all")

does_nothing = PowerLawFitDeriv()
