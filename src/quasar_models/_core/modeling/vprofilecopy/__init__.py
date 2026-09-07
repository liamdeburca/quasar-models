__all__ = [
    "VProfileCopyEvaluate",
    "VProfileCopyFitDeriv",
    "choose_evaluate_func",
    "choose_fit_deriv_func",
    "evaluate_v",
    "fit_deriv_v_all",
    "fit_deriv_v_only_strength_scale",
]
from .utils import VProfileCopyEvaluate, VProfileCopyFitDeriv


def choose_evaluate_func() -> VProfileCopyEvaluate:
    return evaluate_v


def choose_fit_deriv_func(fixed: dict[str, bool]) -> VProfileCopyFitDeriv:
    return fit_deriv_v_all
    # _case = (
    #     fixed.get("strength_scale", False),
    #     all(v for k, v in fixed.items() if k != "strength_scale"),
    # )
    # match _case:
    #     case False, False:
    #         return fit_deriv_v_all
    #     case False, True:
    #         return fit_deriv_v_only_strength_scale
    #     case True, True:
    #         return does_nothing


evaluate_v = VProfileCopyEvaluate("evaluate_v")

fit_deriv_v_all = VProfileCopyFitDeriv("fit_deriv_v_all")
fit_deriv_v_only_strength_scale = VProfileCopyFitDeriv(
    "fit_deriv_v_only_strength_scale"
)

does_nothing = VProfileCopyFitDeriv()
