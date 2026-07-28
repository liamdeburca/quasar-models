__all__ = [
    "IronEvaluate",
    "IronFitDeriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
    "evaluate_exact",
    "evaluate_exact_no_split",
    "evaluate_interp",
    "fit_deriv_exact_all",
    "fit_deriv_exact_all_except_flux",
    "fit_deriv_exact_all_except_fwhm",
    "fit_deriv_exact_all_except_left",
    "fit_deriv_exact_all_except_right",
    "fit_deriv_exact_all_except_split",
    "fit_deriv_exact_flux_and_fwhm",
    "fit_deriv_exact_flux_and_fwhm_and_left",
    "fit_deriv_exact_flux_and_fwhm_and_right",
    "fit_deriv_exact_flux_and_fwhm_and_split",
    "fit_deriv_exact_flux_and_left",
    "fit_deriv_exact_flux_and_left_and_right",
    "fit_deriv_exact_flux_and_right",
    "fit_deriv_exact_flux_and_split",
    "fit_deriv_exact_flux_and_split_and_left",
    "fit_deriv_exact_flux_and_split_and_right",
    "fit_deriv_exact_fwhm_and_left",
    "fit_deriv_exact_fwhm_and_left_and_right",
    "fit_deriv_exact_fwhm_and_right",
    "fit_deriv_exact_fwhm_and_split",
    "fit_deriv_exact_fwhm_and_split_and_left",
    "fit_deriv_exact_fwhm_and_split_and_right",
    "fit_deriv_exact_left_and_right",
    "fit_deriv_exact_no_split_all",
    "fit_deriv_exact_no_split_only_flux",
    "fit_deriv_exact_no_split_only_fwhm",
    "fit_deriv_exact_only_flux",
    "fit_deriv_exact_only_fwhm",
    "fit_deriv_exact_only_left",
    "fit_deriv_exact_only_right",
    "fit_deriv_exact_only_split",
    "fit_deriv_exact_split_and_left",
    "fit_deriv_exact_split_and_left_and_right",
    "fit_deriv_exact_split_and_right",
    "fit_deriv_interp_all",
    "fit_deriv_interp_only_flux",
    "fit_deriv_interp_only_fwhm",
]
from .utils import (
    IronEvaluate,
    IronFitDeriv,
)


def choose_evaluate_func(
    left: float,
    right: float,
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> IronEvaluate:
    if all(
        [
            left == 1.0,
            right == 1.0,
            fixed.get("left", False),
            fixed.get("right", False),
        ]
    ):
        return evaluate_interp if allow_interp_fitting else evaluate_exact_no_split
    else:
        return evaluate_exact


def choose_fit_deriv_func(
    left: float,
    right: float,
    allow_interp_fitting: bool,
    fixed: dict[str, bool],
) -> IronFitDeriv:
    if all(
        [
            left == 1.0,
            right == 1.0,
            fixed.get("left", False),
            fixed.get("right", False),
        ]
    ):
        _case = (fixed.get("flux", False), fixed.get("fwhm", False))
        if allow_interp_fitting:
            match _case:
                case False, False:
                    return fit_deriv_interp_all
                case False, True:
                    return fit_deriv_interp_only_flux
                case True, False:
                    return fit_deriv_interp_only_fwhm
                case True, True:
                    return does_nothing
        else:
            match _case:
                case False, False:
                    return fit_deriv_exact_no_split_all
                case False, True:
                    return fit_deriv_exact_no_split_only_flux
                case True, False:
                    return fit_deriv_exact_no_split_only_fwhm
                case True, True:
                    return does_nothing
    else:
        _case = (
            fixed.get("flux", False),
            fixed.get("fwhm", False),
            fixed.get("split", False),
            fixed.get("left", False),
            fixed.get("right", False),
        )
        match _case:
            # General case
            case False, False, False, False, False:
                return fit_deriv_exact_all

            # All except one
            case True, False, False, False, False:
                return fit_deriv_exact_all_except_flux
            case False, True, False, False, False:
                return fit_deriv_exact_all_except_fwhm
            case False, False, True, False, False:
                return fit_deriv_exact_all_except_split
            case False, False, False, True, False:
                return fit_deriv_exact_all_except_left
            case False, False, False, False, True:
                return fit_deriv_exact_all_except_right

            # Three free
            case True, True, False, False, False:
                return fit_deriv_exact_split_and_left_and_right
            case True, False, True, False, False:
                return fit_deriv_exact_fwhm_and_left_and_right
            case True, False, False, True, False:
                return fit_deriv_exact_fwhm_and_split_and_right
            case True, False, False, False, True:
                return fit_deriv_exact_fwhm_and_split_and_left
            case False, True, True, False, False:
                return fit_deriv_exact_flux_and_left_and_right
            case False, True, False, True, False:
                return fit_deriv_exact_flux_and_split_and_right
            case False, True, False, False, True:
                return fit_deriv_exact_flux_and_split_and_left
            case False, False, True, True, False:
                return fit_deriv_exact_flux_and_fwhm_and_right
            case False, False, True, False, True:
                return fit_deriv_exact_flux_and_fwhm_and_left
            case False, False, False, True, True:
                return fit_deriv_exact_flux_and_fwhm_and_split

            # Two free
            case True, True, True, False, False:
                return fit_deriv_exact_left_and_right
            case True, True, False, True, False:
                return fit_deriv_exact_split_and_right
            case True, True, False, False, True:
                return fit_deriv_exact_split_and_left
            case True, False, True, True, False:
                return fit_deriv_exact_fwhm_and_right
            case True, False, True, False, True:
                return fit_deriv_exact_fwhm_and_left
            case True, False, False, True, True:
                return fit_deriv_exact_fwhm_and_split
            case False, True, True, True, False:
                return fit_deriv_exact_flux_and_right
            case False, True, True, False, True:
                return fit_deriv_exact_flux_and_left
            case False, True, False, True, True:
                return fit_deriv_exact_flux_and_split
            case False, False, True, True, True:
                return fit_deriv_exact_flux_and_fwhm

            # Exactly one free
            case True, True, True, True, False:
                return fit_deriv_exact_only_right
            case True, True, True, False, True:
                return fit_deriv_exact_only_left
            case True, True, False, True, True:
                return fit_deriv_exact_only_split
            case True, False, True, True, True:
                return fit_deriv_exact_only_fwhm
            case False, True, True, True, True:
                return fit_deriv_exact_only_flux

            case True, True, True, True, True:
                return does_nothing


### By CONVOLUTION // No split

evaluate_exact_no_split = IronEvaluate("evaluate_exact_no_split")

fit_deriv_exact_no_split_only_flux = IronFitDeriv("fit_deriv_exact_no_split_only_flux")
fit_deriv_exact_no_split_only_fwhm = IronFitDeriv("fit_deriv_exact_no_split_only_fwhm")
fit_deriv_exact_no_split_all = IronFitDeriv("fit_deriv_exact_no_split_all")

### By CONVOLUTION // With split

evaluate_exact = IronEvaluate("evaluate_exact")

fit_deriv_exact_only_flux = IronFitDeriv("fit_deriv_exact_only_flux")
fit_deriv_exact_only_fwhm = IronFitDeriv("fit_deriv_exact_only_fwhm")
fit_deriv_exact_only_split = IronFitDeriv("fit_deriv_exact_only_split")
fit_deriv_exact_only_left = IronFitDeriv("fit_deriv_exact_only_left")
fit_deriv_exact_only_right = IronFitDeriv("fit_deriv_exact_only_right")

fit_deriv_exact_flux_and_fwhm = IronFitDeriv("fit_deriv_exact_flux_and_fwhm")
fit_deriv_exact_flux_and_split = IronFitDeriv("fit_deriv_exact_flux_and_split")
fit_deriv_exact_flux_and_left = IronFitDeriv("fit_deriv_exact_flux_and_left")
fit_deriv_exact_flux_and_right = IronFitDeriv("fit_deriv_exact_flux_and_right")
fit_deriv_exact_fwhm_and_split = IronFitDeriv("fit_deriv_exact_fwhm_and_split")
fit_deriv_exact_fwhm_and_left = IronFitDeriv("fit_deriv_exact_fwhm_and_left")
fit_deriv_exact_fwhm_and_right = IronFitDeriv("fit_deriv_exact_fwhm_and_right")
fit_deriv_exact_split_and_left = IronFitDeriv("fit_deriv_exact_split_and_left")
fit_deriv_exact_split_and_right = IronFitDeriv("fit_deriv_exact_split_and_right")
fit_deriv_exact_left_and_right = IronFitDeriv("fit_deriv_exact_left_and_right")

fit_deriv_exact_flux_and_fwhm_and_split = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm_and_split"
)
fit_deriv_exact_flux_and_fwhm_and_left = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm_and_left"
)
fit_deriv_exact_flux_and_fwhm_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_fwhm_and_right"
)
fit_deriv_exact_flux_and_split_and_left = IronFitDeriv(
    "fit_deriv_exact_flux_and_split_and_left"
)
fit_deriv_exact_flux_and_split_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_split_and_right"
)
fit_deriv_exact_flux_and_left_and_right = IronFitDeriv(
    "fit_deriv_exact_flux_and_left_and_right"
)
fit_deriv_exact_fwhm_and_split_and_left = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_split_and_left"
)
fit_deriv_exact_fwhm_and_split_and_right = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_split_and_right"
)
fit_deriv_exact_fwhm_and_left_and_right = IronFitDeriv(
    "fit_deriv_exact_fwhm_and_left_and_right"
)
fit_deriv_exact_split_and_left_and_right = IronFitDeriv(
    "fit_deriv_exact_split_and_left_and_right"
)

fit_deriv_exact_all_except_flux = IronFitDeriv("fit_deriv_exact_all_except_flux")
fit_deriv_exact_all_except_fwhm = IronFitDeriv("fit_deriv_exact_all_except_fwhm")
fit_deriv_exact_all_except_split = IronFitDeriv("fit_deriv_exact_all_except_split")
fit_deriv_exact_all_except_left = IronFitDeriv("fit_deriv_exact_all_except_left")
fit_deriv_exact_all_except_right = IronFitDeriv("fit_deriv_exact_all_except_right")

fit_deriv_exact_all = IronFitDeriv("fit_deriv_exact_all")

### By INTERPOLATION

evaluate_interp = IronEvaluate("evaluate_interp")

fit_deriv_interp_only_flux = IronFitDeriv("fit_deriv_interp_only_flux")
fit_deriv_interp_only_fwhm = IronFitDeriv("fit_deriv_interp_only_fwhm")
fit_deriv_interp_all = IronFitDeriv("fit_deriv_interp_all")

does_nothing = IronFitDeriv()
