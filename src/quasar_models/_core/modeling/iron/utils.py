from collections.abc import Callable
from typing import ClassVar, Literal

from quasar_typing.numpy import FloatMatrix, FloatVector

from ..template.cytemplate import CyTemplate
from ..utils import _TemplateEvaluate, _TemplateFitDeriv
from . import evaluate, fit_deriv


class IronEvaluate(_TemplateEvaluate):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        template: object | CyTemplate,
        scale: float,
        n_scales: float,
        interpolation_matrix: tuple | Literal[False] | None = None,
        y: FloatVector | None = None,
    ) -> FloatVector:
        """
        Evaluate an iron model at the given wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        flux : float
            Flux density at ('fwhm_norm', 'x_norm').
        fwhm : float
            FWHM (km/s) of the template.
        split : float
            Wavelength of the split.
        left : float
            Flux scaling factor on the left side of the split.
        right : float
            Flux scaling factor on the right side of the split.
        template : object | CyTemplate
        scale : float
            Scale factor (c) used for the sigmoid function at the split.
        n_scales : float
        interpolation_matrix : tuple, optional
        y : FloatVector, optional
            Flux density array to modify in place.

        Returns
        -------
        y : FloatVector
            Flux density array at the given wavelength array.
        """
        if not isinstance(template, CyTemplate):
            template = CyTemplate.fromTemplate(
                template,
                simplify=self.simplify,
            )

        return super().__call__(
            x,
            template.x,
            (flux, fwhm, split, left, right, template, scale, n_scales),
            interpolation_matrix=interpolation_matrix,
            y=y,
        )


class IronFitDeriv(_TemplateFitDeriv):
    _ndim: ClassVar[Literal[5]] = 5

    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        template: object | CyTemplate,
        scale: float,
        n_scales: float,
        interpolation_matrix: tuple | Literal[False] | None = None,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        """
        Evaluate an iron model at the given wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        flux : float
            Flux density at ('fwhm_norm', 'x_norm').
        fwhm : float
            FWHM (km/s) of the template.
        split : float
            Wavelength of the split.
        left : float
            Flux scaling factor on the left side of the split.
        right : float
            Flux scaling factor on the right side of the split.
        template : object | CyTemplate
        scale : float
            Scale factor (c) used for the sigmoid function at the split.
        n_scales : float
        interpolation_matrix : tuple, optional
        derivs : FloatMatrix, optional
            Partial derivatives array to modify in place.

        Returns
        -------
        derivs : FloatMatrix
            Partial derivatives at the given wavelength array.
        """
        if not isinstance(template, CyTemplate):
            template = CyTemplate.fromTemplate(
                template, 
                simplify=self.simplify,
            )

        return super().__call__(
            x,
            template.x,
            (flux, fwhm, split, left, right, template, scale, n_scales),
            interpolation_matrix=interpolation_matrix,
            derivs=derivs,
        )