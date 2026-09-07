__all__ = [
    "LinearTie",
    "Param",
    "chain_linear_ties",
]

from collections.abc import Iterable
from dataclasses import dataclass
from math import inf
from typing import ClassVar, Literal, Self

from astropy.modeling import CompoundModel, Parameter

from .base_model import BaseModel


@dataclass(kw_only=True, frozen=True)
class LinearTie:
    a: float
    b: float

    model_name: str
    parameter_name: str

    def __bool__(self) -> Literal[True]:
        return True

    def __call__(self, model: CompoundModel) -> float:
        x = getattr(model[self.model_name], self.parameter_name).value
        return self.a * x + self.b

    def copy(self) -> Self:
        return LinearTie(
            a=self.a,
            b=self.b,
            model_name=self.model_name,
            parameter_name=self.parameter_name,
        )

@dataclass(kw_only=True, frozen=True)
class IdenticalTie(LinearTie):
    model_name: str
    parameter_name: str

    a: ClassVar[Literal[1]] = 1.0
    b: ClassVar[Literal[0]] = 0.0


def chain_linear_ties(*ties: LinearTie) -> LinearTie:
    if len(ties) < 2:
        raise ValueError("At least two LinearTie instances are required to chain.")

    tie_as = (tie.a for tie in ties)
    tie_bs = (tie.b for tie in ties)

    new_a: float = 1.0
    new_b: float = 0.0
    for a, b in zip(tie_as, tie_bs):
        new_b += new_a * b
        new_a *= a

    return LinearTie(
        a=new_a, 
        b=new_b, 
        model_name=ties[-1].model_name, 
        parameter_name=ties[-1].parameter_name,
    )

@dataclass
class Param:
    name: str
    value: float
    bounds: tuple[float, float]

    tied: LinearTie | None
    fixed: bool

    @classmethod
    def from_parameter(cls, param: Self | Parameter) -> Self:
        if isinstance(param, cls): 
            return param
        
        lb, ub = param.bounds
        tied = param.tied
        
        return Param(
            name=param.name,
            value=param.value,
            bounds=(-inf if lb is None else lb, inf if ub is None else ub),
            tied=tied if isinstance(tied, LinearTie) else None,
            fixed=param.fixed,
        )

# Parameter classification

def get_param_type(
    param: Param,
    submodels: Iterable[BaseModel],
    *,
    coerce: bool,
) -> Literal['free', 'tied', 'fixed']:
    """
    Classifies a parameter as 'free', 'tied', or 'fixed'.

    To be classified as 'fixed', the parameter must either be fixed itself or be 
    tied to a fixed parameter.

    To be classified as 'tied', the parameter must be tied to another parameter
    that is not fixed. If the parameter is tied to a chain of parameters, it will
    be classified as 'tied' if the last parameter in the chain is not fixed.

    To be classified as 'free', the parameter must fail both the 'fixed' and 
    'tied' checks.

    Raises
    ------
    ValueError
        If the parameter is tied to a non-existent parameter, and `coerce` is 
        False.
    """
    target_param: Param = param
    _submodels: list[BaseModel] = list(submodels)
    tie_chain: list[LinearTie] = []

    while True:
        # Tied to a fixed parameter
        if target_param.fixed:
            if coerce: 
                # If the target parameter is fixed:
                # - Fix the root parameter.
                # - Remove the tie from the root parameter.
                param.fixed = True
                param.tied = None
            return 'fixed'

        tied = target_param.tied
        if tied is None:
            break

        # Tied to a tied parameter
        tie_chain.append(tied)
        target_submodels = list(filter(
            lambda m: m.name == tied.model_name,
            _submodels,
        ))
        n = len(target_submodels)
        if n > 1:
            msg = f"Found {n} submodels with name '{tied.model_name}'"
            raise ValueError(msg)
        elif n == 0:
            if coerce:
                # If the target parameter is tied to a parameter that does not exist:
                # - Fix the root parameter.
                # - Remove the tie from the root parameter.
                param.fixed = True
                param.tied = None
                return 'fixed'
            
            msg = f"No submodel found with name '{tied.model_name}'"
            raise ValueError(msg)

        target_submodel = target_submodels[0]
        _submodels.remove(target_submodel)
        try:
            target_param = Param.from_parameter(getattr(
                target_submodel,
                tied.parameter_name,
            ))
        except AttributeError as e:
            msg = f"Submodel '{tied.model_name}' does not have a parameter named '{tied.parameter_name}'"
            raise ValueError(msg) from e

    n_ties: int = len(tie_chain)
    if n_ties == 0:
        # The parameter is a simple free parameter
        return 'free'

    if coerce and n_ties > 1:
        # If the parameter is tied to a chain of parameters:
        # - Collapse the chain into a single tie.
        # - Update the parameter's tie to the new tie.
        param.tied = chain_linear_ties(*tie_chain)

    return 'tied'