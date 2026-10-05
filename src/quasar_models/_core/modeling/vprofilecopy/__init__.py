from typing import Literal, overload, final

from .utils import (
    VProfileCopyEvaluateV,
    VProfileCopyEvaluateX,
    VProfileCopyFitDerivV,
    VProfileCopyFitDerivX,
)

### Types
VProfileCopyEvaluate = VProfileCopyEvaluateV | VProfileCopyEvaluateX
VProfileCopyFitDeriv = VProfileCopyFitDerivV | VProfileCopyFitDerivX


@overload
def choose_evaluate_func(logbinned: Literal[True]) -> VProfileCopyEvaluateV: ...

@overload
def choose_evaluate_func(logbinned: Literal[False]) -> VProfileCopyEvaluateX: ...

@final
def choose_evaluate_func(
    logbinned: bool,
) -> VProfileCopyEvaluateV | VProfileCopyEvaluateX:
    return evaluate_v if logbinned else evaluate_x


@overload
def choose_fit_deriv_func(
    logbinned: Literal[True], 
    fixed: dict[str, bool],
) -> VProfileCopyFitDerivV: ...

@overload
def choose_fit_deriv_func(
    logbinned: Literal[False], 
    fixed: dict[str, bool],
) -> VProfileCopyFitDerivX: ...

@final
def choose_fit_deriv_func(
    logbinned: bool, 
    fixed: dict[str, bool],
) -> VProfileCopyFitDerivV | VProfileCopyFitDerivX:
    _case = (
        bool(fixed["strength_scale"]),
        bool(fixed["strength_1"])
    )
    if logbinned:
        match _case:
            case (False, False) | (True, False):
                return fit_deriv_v_all
            case (False, True):
                return fit_deriv_v_only_strength_scale
            case (True, True):
                return does_nothing_v
    else:
        match _case:
            case (False, False) | (True, False):
                return fit_deriv_x_all
            case (False, True):
                return fit_deriv_x_only_strength_scale
            case (True, True):
                return does_nothing_x


evaluate_v = VProfileCopyEvaluateV("evaluate_v")
evaluate_x = VProfileCopyEvaluateX("evaluate_x")

fit_deriv_v_all = VProfileCopyFitDerivV("fit_deriv_v_all")
fit_deriv_v_only_strength_scale = VProfileCopyFitDerivV(
    "fit_deriv_v_only_strength_scale"
)

fit_deriv_x_all = VProfileCopyFitDerivX("fit_deriv_x_all")
fit_deriv_x_only_strength_scale = VProfileCopyFitDerivX(
    "fit_deriv_x_only_strength_scale"
)

does_nothing_v = VProfileCopyFitDerivV()
does_nothing_x = VProfileCopyFitDerivX()
