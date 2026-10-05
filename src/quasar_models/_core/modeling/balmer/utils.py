from collections.abc import Callable
from typing import ClassVar, Literal

from quasar_typing.numpy import FloatMatrix, FloatVector

from ..template.cytemplate import CyTemplate
from ..utils import (
    _TemplateEvaluate,
    _TemplateFitDeriv,
)
from . import evaluate, fit_deriv


class BalmerEvaluate(_TemplateEvaluate):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        continuum_template: object | CyTemplate,
        series_template: object | CyTemplate,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
        y: FloatVector | None = None,
    ) -> FloatVector:
        """
        Evaluate a Balmer model at the given wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        flux : float
            Flux density at ('fwhm_norm', 'x_norm').
        fwhm : float
            FWHM (km/s) of the template.
        ratio : float
            Cont./seri. flux ratio 
        continuum_template : object | CyTemplate
            Continuum template object.
        series_template : object | CyTemplate
            Series template object.
        n_scales : float
        interpolation_matrix : tuple, optional
        y : FloatVector, optional
            Existing flux density array to modify in place.

        Returns
        -------
        y : FloatVector
            Flux density array.
        """
        if not isinstance(continuum_template, CyTemplate):
            continuum_template = CyTemplate.fromTemplate(
                continuum_template,
                simplify=self.simplify,
            )
        if not isinstance(series_template, CyTemplate):
            series_template = CyTemplate.fromTemplate(
                series_template,
                simplify=self.simplify,
            )

        return super().__call__(
            x,
            continuum_template.x,
            (flux, fwhm, ratio, continuum_template, series_template, n_scales),
            interpolation_matrix=interpolation_matrix,
            y=y,
        )


class BalmerFitDeriv(_TemplateFitDeriv):
    _ndim: ClassVar[Literal[3]] = 3

    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        continuum_template: object | CyTemplate,
        series_template: object | CyTemplate,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        """
        Calculate the partial derivatives of a Balmer model at the given 
        wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        flux : float
            Flux density at ('fwhm_norm', 'x_norm').
        fwhm : float
            FWHM (km/s) of the template.
        ratio : float
            Cont./seri. flux ratio 
        continuum_template : object | CyTemplate
            Continuum template object.
        series_template : object | CyTemplate
            Series template object.
        n_scales : float
        interpolation_matrix : tuple, optional
        derivs : FloatMatrix, optional
            Existing partial derivatives array to modify in place.

        Returns
        -------
        derivs : FloatMatrix
            Partial derivatives array.
        """
        if not isinstance(continuum_template, CyTemplate):
            continuum_template = CyTemplate.fromTemplate(
                continuum_template,
                simplify=self.simplify,
            )
        if not isinstance(series_template, CyTemplate):
            series_template = CyTemplate.fromTemplate(
                series_template,
                simplify=self.simplify,
            )

        return super().__call__(
            x,
            continuum_template.x,
            (flux, fwhm, ratio, continuum_template, series_template, n_scales),
            interpolation_matrix=interpolation_matrix,
            derivs=derivs,
        )
