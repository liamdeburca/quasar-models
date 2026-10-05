from .utils import (
    PowerLawEvaluate,
    PowerLawFitDeriv,
    PowerLawInverse,
)


def choose_evaluate_func() -> PowerLawEvaluate:
    return evaluate


def choose_fit_deriv_func(fixed: dict[str, bool]) -> PowerLawFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["alpha"]),
    )
    match _case:
        case (False, False):
            return fit_deriv_all
        case (False, True):
            return fit_deriv_only_flux
        case (True, False):
            return fit_deriv_only_alpha
        case (True, True):
            return does_nothing

###

evaluate = PowerLawEvaluate("evaluate")
inverse = PowerLawInverse("inverse")

fit_deriv_only_flux = PowerLawFitDeriv("fit_deriv_only_flux")
fit_deriv_only_alpha = PowerLawFitDeriv("fit_deriv_only_alpha")
fit_deriv_all = PowerLawFitDeriv("fit_deriv_all")

does_nothing = PowerLawFitDeriv()
