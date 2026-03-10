from typing import Self
from astropy.modeling import Parameter

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from . import evaluation
from ...utils.basemodel import BaseModel

class BalmerAttenuationModel(BaseModel):
    tau   = Parameter(default=1.0)
    scale = Parameter(default=3.0)

    def __init__(
        self,
        tau: float,
        scale: float,
        *,
        edge: float = None,
        name: str = 'balmer_attenuation',
        **kwargs,
    ):
        assert edge is not None, "edge must be provided"

        super().__init__(
            tau, scale, 
            name=name, **kwargs,
        )
        self.edge: float = edge

    def evaluate(self, x, tau, scale):
        return evaluation.evaluate(
            x,
            tau, scale,
            edge=self.edge,
        )

    def fit_deriv(self, x, tau, scale):
        return evaluation.fit_deriv(
            x,
            tau, scale,
            edge=self.edge, 
            fixed=self.fixed,
        )
    
    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, BalmerAttenuationModel):
            msg = f"Expected a BalmerAttenuationModel instance, \
                got {type(value).__name__} instead"
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    def _ignore(self) -> None:
        """
        Ignore the Balmer attenuation by nullifying its contribution.
        """
        assert not hasattr(self, '_old_specs'), "this model is already ignored"

        self._old_specs: dict[str, Parameter] = {
            'tau':   self.tau.copy(),
            'scale': self.scale.copy(),
        }

        self.tau.value   = 0.0
        self.tau.fixed   = True
        self.tau.bounds  = None
        self.scale.fixed = True

    def _restore(self) -> None:
        """
        Restore the Balmer attenuation model to its previous state.
        """
        assert hasattr(self, '_old_specs'), "this model is not ignored"

        self.tau   = self._old_specs['tau']
        self.scale = self._old_specs['scale']

        del self._old_specs