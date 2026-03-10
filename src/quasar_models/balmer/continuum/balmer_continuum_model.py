from typing import Self
from astropy.modeling import Parameter

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from . import evaluation
from ...utils.basemodel import BaseModel

class BalmerContinuumModel(BaseModel):
    flux = Parameter(default=1.0, min=0.0)
    temp = Parameter(default=15_000.0, fixed=True)

    def __init__(
        self,
        flux: float,
        temp: float,
        *,
        edge: float | None = None,
        boltz: float | None = None,
        name: str = 'balmer_continuum',
        **kwargs,
    ):
        assert edge is not None, "edge must be provided"
        assert boltz is not None, "boltz must be provided"

        super().__init__(
            flux,
            temp,
            name=name,
            **kwargs,
        )
        self.edge:  float = edge
        self.boltz: float = boltz

    def evaluate(self, x, flux, temp):
        """
        Evaluate the Balmer continuum model.
        
        Parameters
        ----------
        x : array-like
            Dimensionless wavelength array.
        flux : float
            Dimensionless flux of the continuum at the Balmer edge.
        temp : float
            Dimensionless temperature parameter for the Balmer continuum model.
        
        Returns
        -------
        f : array_like
            The evaluated continuum model values.
        Raises
        ------
        AssertionError
            If the Boltzmann constant (self.boltz) has not been initialized.
        """
        return evaluation.evaluate(
            x, 
            flux, temp, 
            edge=self.edge, boltz=self.boltz,
        )

    def fit_deriv(self, x, flux, temp):
        """
        Calculate the derivative of the fitting function with respect to parameters.
        This method computes the partial derivatives of the continuum model fit
        with respect to the fitting parameters, used for optimization algorithms.
        
        Parameters
        ----------
        x : array-like
            Dimensionless wavelength array.
        flux : float
            Dimensionless flux of the continuum at the Balmer edge.
        temp : float
            Dimensionless temperature parameter for the Balmer continuum model.
        
        Returns
        -------
        df : list of array-like
            Derivative arrays of the model with respect to the fitting 
            parameters, evaluated at the given x positions.
        
        Raises
        ------
        AssertionError
            If the Boltzmann constant (self.boltz) has not been initialized.
        """
        return evaluation.fit_deriv(
            x,
            flux, temp,
            edge=self.edge, boltz=self.boltz,
            fixed=self.fixed,
        )

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, BalmerContinuumModel):
            msg = f"Expected a BalmerContinuumModel instance, \
                got {type(value)} instead."
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    def _ignore(self) -> None:
        """
        Ignore the Balmer continuum model by nullifying its contribution.
        """
        assert not hasattr(self, '_old_specs'), "this model is already ignored"

        self._old_specs: dict[str, Parameter] = {
            'flux': self.flux.copy(),
            'temp': self.temp.copy(),
        }
        
        self.flux.value  = 0.0
        self.flux.fixed  = True
        self.flux.bounds = None
        self.temp.fixed  = True

    def _restore(self) -> None:
        """
        Restore the Balmer continuum model to its previous state before 
        ignoring.
        """
        assert hasattr(self, '_old_specs'), "this model was not ignored"

        self.flux = self._old_specs['flux']
        self.temp = self._old_specs['temp']
        
        del self._old_specs