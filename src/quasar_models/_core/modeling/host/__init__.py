from .utils import (
    HostGalaxyEvaluate,
    HostGalaxyFitDeriv,
)


def choose_evaluate_func(
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> HostGalaxyEvaluate:
    if fixed["fwhm"]:
        return evaluate_rescale
    elif allow_interp_fitting:
        return evaluate_interp
    else:
        return evaluate_exact


def choose_fit_deriv_rescale_func(
    fixed: dict[str, bool],
) -> HostGalaxyFitDeriv:
    assert fixed["fwhm"]
    return does_nothing \
        if fixed["flux"] \
        else fit_deriv_rescale_only_flux


def choose_fit_deriv_interp_func(
    fixed: dict[str, bool],
) -> HostGalaxyFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["fwhm"]),
    )
    match _case:
        case (False, False):
            return fit_deriv_interp_all
        case (False, True):
            return fit_deriv_interp_only_flux
        case (True, False):
            return fit_deriv_interp_only_fwhm
        case (True, True):
            return does_nothing


def choose_fit_deriv_exact_func(
    fixed: dict[str, bool],
) -> HostGalaxyFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["fwhm"]),
    )
    match _case:
        case (False, False):
            return fit_deriv_exact_all
        case (False, True):
            return fit_deriv_exact_only_flux
        case (True, False):
            return fit_deriv_exact_only_fwhm
        case (True, True):
            return does_nothing


def choose_fit_deriv_func(
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> HostGalaxyFitDeriv:

    if fixed["fwhm"]:
        return choose_fit_deriv_rescale_func(fixed)
    elif allow_interp_fitting:
        return choose_fit_deriv_interp_func(fixed)
    else:
        return choose_fit_deriv_exact_func(fixed)


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

### By RESCALING

evaluate_rescale = HostGalaxyEvaluate(
    "evaluate_rescale",
    simplify=True,
    rescaling=True,
)

fit_deriv_rescale_only_flux = HostGalaxyFitDeriv(
    "fit_deriv_rescale_only_flux",
    simplify=True,
    rescaling=True,
)

does_nothing = HostGalaxyFitDeriv()
