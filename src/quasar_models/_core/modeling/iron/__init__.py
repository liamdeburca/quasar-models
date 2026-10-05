from .utils import (
    IronEvaluate,
    IronFitDeriv,
)


def _no_split(left: float, right: float, fixed: dict[str, bool]) -> bool:
    return left == 1.0 and right == 1.0 and fixed["left"] and fixed["right"]


def choose_evaluate_func(
    left: float,
    right: float,
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> IronEvaluate:
    if _no_split(left, right, fixed):
        if fixed["fwhm"]:
            return evaluate_rescale
        elif allow_interp_fitting:
            return evaluate_interp
        return evaluate_exact_no_split
    return evaluate_exact


def choose_fit_deriv_rescale_func(
    fixed: dict[str, bool],
) -> IronFitDeriv:
    if fixed["fwhm"]:
        return fit_deriv_rescale_only_flux
    return does_nothing


def choose_fit_deriv_interp_func(
    fixed: dict[str, bool],
) -> IronFitDeriv:
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


def choose_fit_deriv_exact_no_split_func(
    fixed: dict[str, bool],
) -> IronFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["fwhm"]),
    )
    match _case:
        case (False, False):
            return fit_deriv_exact_no_split_all
        case (False, True):
            return fit_deriv_exact_no_split_only_flux
        case (True, False):
            return fit_deriv_exact_no_split_only_fwhm
        case (True, True):
            return does_nothing


def choose_fit_deriv_exact_func(
    fixed: dict[str, bool],
) -> IronFitDeriv:
    _case = (
        bool(fixed["flux"]),
        bool(fixed["fwhm"]),
        bool(fixed["split"]),
        bool(fixed["left"]),
        bool(fixed["right"]),
    )
    match _case:
        # General case
        case (False, False, False, False, False):
            return fit_deriv_exact_all

        # All except one
        case (True, False, False, False, False):
            return fit_deriv_exact_all_except_flux
        case (False, True, False, False, False):
            return fit_deriv_exact_all_except_fwhm
        case (False, False, True, False, False):
            return fit_deriv_exact_all_except_split
        case (False, False, False, True, False):
            return fit_deriv_exact_all_except_left
        case (False, False, False, False, True):
            return fit_deriv_exact_all_except_right

        # Three free
        case (True, True, False, False, False):
            return fit_deriv_exact_split_and_left_and_right
        case (True, False, True, False, False):
            return fit_deriv_exact_fwhm_and_left_and_right
        case (True, False, False, True, False):
            return fit_deriv_exact_fwhm_and_split_and_right
        case (True, False, False, False, True):
            return fit_deriv_exact_fwhm_and_split_and_left
        case (False, True, True, False, False):
            return fit_deriv_exact_flux_and_left_and_right
        case (False, True, False, True, False):
            return fit_deriv_exact_flux_and_split_and_right
        case (False, True, False, False, True):
            return fit_deriv_exact_flux_and_split_and_left
        case (False, False, True, True, False):
            return fit_deriv_exact_flux_and_fwhm_and_right
        case (False, False, True, False, True):
            return fit_deriv_exact_flux_and_fwhm_and_left
        case (False, False, False, True, True):
            return fit_deriv_exact_flux_and_fwhm_and_split

        # Two free
        case (True, True, True, False, False):
            return fit_deriv_exact_left_and_right
        case (True, True, False, True, False):
            return fit_deriv_exact_split_and_right
        case (True, True, False, False, True):
            return fit_deriv_exact_split_and_left
        case (True, False, True, True, False):
            return fit_deriv_exact_fwhm_and_right
        case (True, False, True, False, True):
            return fit_deriv_exact_fwhm_and_left
        case (True, False, False, True, True):
            return fit_deriv_exact_fwhm_and_split
        case (False, True, True, True, False):
            return fit_deriv_exact_flux_and_right
        case (False, True, True, False, True):
            return fit_deriv_exact_flux_and_left
        case (False, True, False, True, True):
            return fit_deriv_exact_flux_and_split
        case (False, False, True, True, True):
            return fit_deriv_exact_flux_and_fwhm

        # Exactly one free
        case (True, True, True, True, False):
            return fit_deriv_exact_only_right
        case (True, True, True, False, True):
            return fit_deriv_exact_only_left
        case (True, True, False, True, True):
            return fit_deriv_exact_only_split
        case (True, False, True, True, True):
            return fit_deriv_exact_only_fwhm
        case (False, True, True, True, True):
            return fit_deriv_exact_only_flux

        case (True, True, True, True, True):
            return does_nothing

def choose_fit_deriv_func(
    left: float,
    right: float,
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> IronFitDeriv:
    if _no_split(left, right, fixed):
        if fixed["fwhm"]:
            return choose_fit_deriv_rescale_func(fixed)
        elif allow_interp_fitting:
            return choose_fit_deriv_interp_func(fixed)
        return choose_fit_deriv_exact_no_split_func(fixed)
    return choose_fit_deriv_exact_func(fixed)

### By CONVOLUTION // No split: Simplify template

evaluate_exact_no_split = IronEvaluate(
    "evaluate_exact_no_split",
    simplify=True,
)

fit_deriv_exact_no_split_only_flux = IronFitDeriv(
    "fit_deriv_exact_no_split_only_flux",
    simplify=True,
)
fit_deriv_exact_no_split_only_fwhm = IronFitDeriv(
    "fit_deriv_exact_no_split_only_fwhm",
    simplify=True,
)
fit_deriv_exact_no_split_all = IronFitDeriv(
    "fit_deriv_exact_no_split_all",
    simplify=True,
)

### By CONVOLUTION // With split: Simplify template

evaluate_exact = IronEvaluate(
    "evaluate_exact",
    simplify=True,
)

fit_deriv_exact_only_flux = IronFitDeriv(
    "fit_deriv_exact_only_flux",
    simplify=True,
)
fit_deriv_exact_only_fwhm = IronFitDeriv(
    "fit_deriv_exact_only_fwhm",
    simplify=True,
)
fit_deriv_exact_only_split = IronFitDeriv(
    "fit_deriv_exact_only_split",
    simplify=True,
)
fit_deriv_exact_only_left = IronFitDeriv(
    "fit_deriv_exact_only_left",
    simplify=True,
)
fit_deriv_exact_only_right = IronFitDeriv(
    "fit_deriv_exact_only_right",
    simplify=True,
)

fit_deriv_exact_flux_and_fwhm = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm",
    simplify=True,
)
fit_deriv_exact_flux_and_split = IronFitDeriv(
    "fit_deriv_exact_flux_and_split",
    simplify=True,
)
fit_deriv_exact_flux_and_left = IronFitDeriv(
    "fit_deriv_exact_flux_and_left",
    simplify=True,
)
fit_deriv_exact_flux_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_right",
    simplify=True,
)
fit_deriv_exact_fwhm_and_split = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_split",
    simplify=True,
)
fit_deriv_exact_fwhm_and_left = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_left",
    simplify=True,
)
fit_deriv_exact_fwhm_and_right = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_right",
    simplify=True,
)
fit_deriv_exact_split_and_left = IronFitDeriv(
    "fit_deriv_exact_split_and_left",
    simplify=True,
)
fit_deriv_exact_split_and_right = IronFitDeriv(
    "fit_deriv_exact_split_and_right",
    simplify=True,
)
fit_deriv_exact_left_and_right = IronFitDeriv(
    "fit_deriv_exact_left_and_right",
    simplify=True,
)

fit_deriv_exact_flux_and_fwhm_and_split = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm_and_split",
    simplify=True,
)
fit_deriv_exact_flux_and_fwhm_and_left = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm_and_left",
    simplify=True,
)
fit_deriv_exact_flux_and_fwhm_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm_and_right",
    simplify=True,
)
fit_deriv_exact_flux_and_split_and_left = IronFitDeriv(
    "fit_deriv_exact_flux_and_split_and_left",
    simplify=True,
)
fit_deriv_exact_flux_and_split_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_split_and_right",
    simplify=True,
)
fit_deriv_exact_flux_and_left_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_left_and_right",
    simplify=True,
)
fit_deriv_exact_fwhm_and_split_and_left = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_split_and_left",
    simplify=True,
)
fit_deriv_exact_fwhm_and_split_and_right = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_split_and_right",
    simplify=True,
)
fit_deriv_exact_fwhm_and_left_and_right = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_left_and_right",
    simplify=True,
)
fit_deriv_exact_split_and_left_and_right = IronFitDeriv(
    "fit_deriv_exact_split_and_left_and_right",
    simplify=True,
)

fit_deriv_exact_all_except_flux = IronFitDeriv(
    "fit_deriv_exact_all_except_flux",
    simplify=True,
)
fit_deriv_exact_all_except_fwhm = IronFitDeriv(
    "fit_deriv_exact_all_except_fwhm",
    simplify=True,
)
fit_deriv_exact_all_except_split = IronFitDeriv(
    "fit_deriv_exact_all_except_split",
    simplify=True,
)
fit_deriv_exact_all_except_left = IronFitDeriv(
    "fit_deriv_exact_all_except_left",
    simplify=True,
)
fit_deriv_exact_all_except_right = IronFitDeriv(
    "fit_deriv_exact_all_except_right",
    simplify=True,
)

fit_deriv_exact_all = IronFitDeriv(
    "fit_deriv_exact_all",
    simplify=True,
)

### By INTERPOLATION: No simplification

evaluate_interp = IronEvaluate(
    "evaluate_interp",
    simplify=False,
)

fit_deriv_interp_only_flux = IronFitDeriv(
    "fit_deriv_interp_only_flux",
    simplify=False,
)
fit_deriv_interp_only_fwhm = IronFitDeriv(
    "fit_deriv_interp_only_fwhm",
    simplify=False,
)
fit_deriv_interp_all = IronFitDeriv(
    "fit_deriv_interp_all",
    simplify=False,
)

### By RESCALING

evaluate_rescale = IronEvaluate(
    "evaluate_rescale",
    simplify=False,
    rescaling=True
)

fit_deriv_rescale_only_flux = IronFitDeriv(
    "fit_deriv_rescale_only_flux",
    simplify=False,
    rescaling=True
)

does_nothing = IronFitDeriv()
