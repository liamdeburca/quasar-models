from collections.abc import Callable
from typing import ClassVar, Literal

from quasar_typing.numpy import FloatMatrix, FloatVector

from ..template.cytemplate import CyTemplate
from ..utils import (
    _TemplateEvaluate,
    _TemplateFitDeriv,
)
from . import evaluate, fit_deriv


class HostGalaxyEvaluate(_TemplateEvaluate):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        *,
        template: object | CyTemplate,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
        y: FloatVector | None = None,
    ) -> FloatVector:
        """
        Evaluate a host galaxy model at the given wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        flux : float
            Flux density at ('fwhm_norm', 'x_norm').
        fwhm : float
            FWHM (km/s) of the template.
        template : object or CyTemplate
            Host galaxy template object.
        n_scales : float
        interpolation_matrix : tuple, optional
            Precomputed interpolation matrix. 
        y : FloatVector, optional
            Output array to store the evaluated host galaxy model. If not 
            provided, a new array will be created.

        Returns
        -------
        y : FloatVector
        """
        if not isinstance(template, CyTemplate):
            template = CyTemplate.fromTemplate(
                template,
                simplify=self.simplify,
            )

        return super().__call__(
            x,
            template.x,
            (flux, fwhm, template, n_scales),
            interpolation_matrix=interpolation_matrix,
            y=y,
        )


class HostGalaxyFitDeriv(_TemplateFitDeriv):
    _ndim: ClassVar[Literal[2]] = 2

    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        *,
        template: object | CyTemplate,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        """
        Calculate the partial derivatives of a host galaxy model.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        flux : float
            Flux density at ('fwhm_norm', 'x_norm').
        fwhm : float
            FWHM (km/s) of the template.
        template : object
            Host galaxy template object.
        n_scales : float
        interpolation_matrix : tuple, optional
            Precomputed interpolation matrix.
        derivs : FloatMatrix, optional
            Output array to store the calculated partial derivatives. If not
            provided, a new array will be created.

        Returns
        -------
        derivs : FloatMatrix
            Calculated partial derivatives of the host galaxy model.
        """
        if not isinstance(template, CyTemplate):
            template = CyTemplate.fromTemplate(
                template,
                simplify=self.simplify,
            )

        return super().__call__(
            x,
            template.x,
            (flux, fwhm, template, n_scales),
            interpolation_matrix=interpolation_matrix,
            derivs=derivs,
        )