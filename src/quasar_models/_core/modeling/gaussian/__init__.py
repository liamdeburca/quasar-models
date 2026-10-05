from typing import Literal, final, overload

from .utils import (
    GaussianEvaluateV,
    GaussianEvaluateX,
    GaussianFitDerivV,
    GaussianFitDerivX,
)

### Types
GaussianEvaluate = GaussianEvaluateV | GaussianEvaluateX
GaussianFitDeriv = GaussianFitDerivV | GaussianFitDerivX


@overload
def choose_evaluate_func(logbinned: Literal[True]) -> GaussianEvaluateV: ...

@overload
def choose_evaluate_func(logbinned: Literal[False]) -> GaussianEvaluateX: ...

@final
def choose_evaluate_func(logbinned: bool) -> GaussianEvaluateV | GaussianEvaluateX:
    return evaluate_v if logbinned else evaluate_x


def choose_fit_deriv_v_func(fixed: dict[str, bool]) -> GaussianFitDerivV:
    match (
        bool(fixed["strength"]),
        bool(fixed["fwhm_v"]),
        bool(fixed["v_off"]),
    ):
        case (False, False, False):
            return fit_deriv_v_all

        case (False, True, True):
            return fit_deriv_v_only_strength
        case (True, False, True):
            return fit_deriv_v_only_fwhm_v
        case (True, True, False):
            return fit_deriv_v_only_v_off

        case (False, False, True):
            return fit_deriv_v_strength_and_fwhm_v
        case (False, True, False):
            return fit_deriv_v_strength_and_v_off
        case (True, False, False):
            return fit_deriv_v_fwhm_v_and_v_off

        case (True, True, True):
            return does_nothing_v


def choose_fit_deriv_x_func(fixed: dict[str, bool]) -> GaussianFitDerivX:
    match (
        bool(fixed["strength"]),
        bool(fixed["fwhm_v"]),
        bool(fixed["v_off"]),
    ):
        case (False, False, False):
            return fit_deriv_x_all

        case (False, True, True):
            return fit_deriv_x_only_strength
        case (True, False, True):
            return fit_deriv_x_only_fwhm_v
        case (True, True, False):
            return fit_deriv_x_only_v_off

        case (False, False, True):
            return fit_deriv_x_strength_and_fwhm_v
        case (False, True, False):
            return fit_deriv_x_strength_and_v_off
        case (True, False, False):
            return fit_deriv_x_fwhm_v_and_v_off

        case (True, True, True):
            return does_nothing_x


@overload
def choose_fit_deriv_func(
    logbinned: Literal[True], 
    fixed: dict[str, bool],
) -> GaussianFitDerivV: ...

@overload
def choose_fit_deriv_func(
    logbinned: Literal[False], 
    fixed: dict[str, bool],
) -> GaussianFitDerivX: ...

@final
def choose_fit_deriv_func(
    logbinned: bool, 
    fixed: dict[str, bool],
) -> GaussianFitDerivV | GaussianFitDerivX:
    if logbinned:
        return choose_fit_deriv_v_func(fixed)
    return choose_fit_deriv_x_func(fixed)

###

evaluate_v = GaussianEvaluateV("evaluate_v")

fit_deriv_v_only_strength = GaussianFitDerivV("fit_deriv_v_only_strength")
fit_deriv_v_only_fwhm_v = GaussianFitDerivV("fit_deriv_v_only_fwhm_v")
fit_deriv_v_only_v_off = GaussianFitDerivV("fit_deriv_v_only_v_off")

fit_deriv_v_strength_and_fwhm_v = GaussianFitDerivV("fit_deriv_v_strength_and_fwhm_v")
fit_deriv_v_strength_and_v_off = GaussianFitDerivV("fit_deriv_v_strength_and_v_off")
fit_deriv_v_fwhm_v_and_v_off = GaussianFitDerivV("fit_deriv_v_fwhm_v_and_v_off")

fit_deriv_v_all = GaussianFitDerivV("fit_deriv_v_all")
does_nothing_v = GaussianFitDerivV()

###

evaluate_x = GaussianEvaluateX("evaluate_x")

fit_deriv_x_only_strength = GaussianFitDerivX("fit_deriv_x_only_strength")
fit_deriv_x_only_fwhm_v = GaussianFitDerivX("fit_deriv_x_only_fwhm_v")
fit_deriv_x_only_v_off = GaussianFitDerivX("fit_deriv_x_only_v_off")

fit_deriv_x_strength_and_fwhm_v = GaussianFitDerivX("fit_deriv_x_strength_and_fwhm_v")
fit_deriv_x_strength_and_v_off = GaussianFitDerivX("fit_deriv_x_strength_and_v_off")
fit_deriv_x_fwhm_v_and_v_off = GaussianFitDerivX("fit_deriv_x_fwhm_v_and_v_off")

fit_deriv_x_all = GaussianFitDerivX("fit_deriv_x_all")
does_nothing_x = GaussianFitDerivX()

###

prime_v = GaussianEvaluateV("prime_v")
prime_x = GaussianEvaluateX("prime_x")
