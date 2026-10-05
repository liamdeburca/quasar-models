from .utils import (
    BalmerEvaluate,
    BalmerFitDeriv,
)


def choose_evaluate_func(
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> BalmerEvaluate:
    if fixed["fwhm"]:
        return evaluate_rescale
    return evaluate_interp if allow_interp_fitting else evaluate_exact


def choose_fit_deriv_rescale_func(
    fixed: dict[str, bool],
) -> BalmerFitDeriv:
    assert fixed["fwhm"]

    _case = (
        bool(fixed["flux"]),
        bool(fixed["ratio"]),
    )
    match _case:
        case (False, False):
            return fit_deriv_rescale_flux_and_ratio
        case (False, True):
            return fit_deriv_rescale_only_flux
        case (True, False):
            return fit_deriv_rescale_only_ratio
        case (True, True):
            return does_nothing


def choose_fit_deriv_interp_func(
    fixed: dict[str, bool],
) -> BalmerFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["fwhm"]),
        bool(fixed["ratio"]),
    )
    match _case:
        case (False, False, False):
            return fit_deriv_interp_all
        case (False, False, True):
            return fit_deriv_interp_flux_and_fwhm
        case (False, True, False):
            return fit_deriv_interp_flux_and_ratio
        case (True, False, False):
            return fit_deriv_interp_fwhm_and_ratio
        case (False, True, True):
            return fit_deriv_interp_only_flux
        case (True, False, True):
            return fit_deriv_interp_only_fwhm
        case (True, True, False):
            return fit_deriv_interp_only_ratio
        case (True, True, True):
            return does_nothing


def choose_fit_deriv_exact_func(
    fixed: dict[str, bool],
) -> BalmerFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["fwhm"]),
        bool(fixed["ratio"]),
    )
    match _case:
        case (False, False, False):
            return fit_deriv_exact_all
        case (False, False, True):
            return fit_deriv_exact_flux_and_fwhm
        case (False, True, False):
            return fit_deriv_exact_flux_and_ratio
        case (True, False, False):
            return fit_deriv_exact_fwhm_and_ratio
        case (False, True, True):
            return fit_deriv_exact_only_flux
        case (True, False, True):
            return fit_deriv_exact_only_fwhm
        case (True, True, False):
            return fit_deriv_exact_only_ratio
        case (True, True, True):
            return does_nothing


def choose_fit_deriv_func(
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> BalmerFitDeriv:

    if fixed["fwhm"]:
        return choose_fit_deriv_rescale_func(fixed)
    elif allow_interp_fitting:
        return choose_fit_deriv_interp_func(fixed)
    else:
        return choose_fit_deriv_exact_func(fixed)    


### By CONVOLUTION: Simplify template

evaluate_exact = BalmerEvaluate(
    "evaluate_exact",
    simplify=True,
)

fit_deriv_exact_only_flux = BalmerFitDeriv(
    "fit_deriv_exact_only_flux",
    simplify=True,
)
fit_deriv_exact_only_fwhm = BalmerFitDeriv(
    "fit_deriv_exact_only_fwhm",
    simplify=True,
)
fit_deriv_exact_only_ratio = BalmerFitDeriv(
    "fit_deriv_exact_only_ratio",
    simplify=True,
)

fit_deriv_exact_flux_and_fwhm = BalmerFitDeriv(
    "fit_deriv_exact_flux_and_fwhm",
    simplify=True,
)
fit_deriv_exact_flux_and_ratio = BalmerFitDeriv(
    "fit_deriv_exact_flux_and_ratio",
    simplify=True,
)
fit_deriv_exact_fwhm_and_ratio = BalmerFitDeriv(
    "fit_deriv_exact_fwhm_and_ratio",
    simplify=True,
)

fit_deriv_exact_all = BalmerFitDeriv(
    "fit_deriv_exact_all",
    simplify=True,
)

### By INTERPOLATION: No simplification

evaluate_interp = BalmerEvaluate(
    "evaluate_interp",
    simplify=False,
) 

fit_deriv_interp_only_flux = BalmerFitDeriv(
    "fit_deriv_interp_only_flux",
    simplify=False,
)
fit_deriv_interp_only_fwhm = BalmerFitDeriv(
    "fit_deriv_interp_only_fwhm",
    simplify=False,
)
fit_deriv_interp_only_ratio = BalmerFitDeriv(
    "fit_deriv_interp_only_ratio",
    simplify=False,
)

fit_deriv_interp_flux_and_fwhm = BalmerFitDeriv(
    "fit_deriv_interp_flux_and_fwhm",
    simplify=False,
)
fit_deriv_interp_flux_and_ratio = BalmerFitDeriv(
    "fit_deriv_interp_flux_and_ratio",
    simplify=False,
)
fit_deriv_interp_fwhm_and_ratio = BalmerFitDeriv(
    "fit_deriv_interp_fwhm_and_ratio",
    simplify=False,
)
fit_deriv_interp_all = BalmerFitDeriv(
    "fit_deriv_interp_all",
    simplify=False,
)

### By RESCALING: Simplification

evaluate_rescale = BalmerEvaluate(
    "evaluate_rescale",
    simplify=True,
    rescaling=True,
)

fit_deriv_rescale_only_flux = BalmerFitDeriv(
    "fit_deriv_rescale_only_flux",
    simplify=True,
    rescaling=True,
)
fit_deriv_rescale_only_ratio = BalmerFitDeriv(
    "fit_deriv_rescale_only_ratio",
    simplify=True,
    rescaling=True,
)
fit_deriv_rescale_flux_and_ratio = BalmerFitDeriv(
    "fit_deriv_rescale_flux_and_ratio",
    simplify=True,
    rescaling=True,
)

does_nothing = BalmerFitDeriv()
