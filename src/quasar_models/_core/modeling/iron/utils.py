from collections.abc import Callable

from numpy import add, float64, zeros
from quasar_typing.numpy import FloatMatrix, FloatVector

from ..template.cytemplate import CyTemplate
from ..utils import _interp, _interp2d, _interp2d_matrix, _interp_matrix
from . import evaluate, fit_deriv


class _IronBase:
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable:
        raise NotImplementedError

    def __init__(
        self,
        func_name: str | None = None,
        simplify: bool = False,
    ) -> None:
        self.func_name: str | None = func_name
        self.__wrapped__: Callable | None = self._get_wrapped(func_name)
        self.simplify: bool = simplify

    def __getstate__(self) -> dict:
        return {
            "func_name": self.func_name,
            "simplify": self.simplify,
        }

    def __setstate__(self, state: dict) -> None:
        self.func_name = state["func_name"]
        self.simplify = state["simplify"]
        self.__wrapped__ = self._get_wrapped(self.func_name)


class IronEvaluate(_IronBase):
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
        template: object,
        cytemplate: CyTemplate | None = None,
        scale: float,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
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
        template : object
        cytemplate : CyTemplate, optional
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
        if cytemplate is None:
            cytemplate = CyTemplate.fromTemplate(
                template,
                simplify=self.simplify,
            )

        _y = zeros(template.x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _y,
                flux,
                fwhm,
                split,
                left,
                right,
                cytemplate,
                scale,
                n_scales,
            )

        if interpolation_matrix is None:
            _y = _interp(x, template.x, _y)
        else:
            _y = _interp_matrix(_y, interpolation_matrix)

        return _y if y is None else add(y, _y, out=y)


class IronFitDeriv(_IronBase):
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
        cytemplate: CyTemplate | None = None,
        template: object,
        scale: float,
        n_scales: float,
        interpolation_matrix: tuple | None = None,
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
        template : object
        cytemplate : CyTemplate, optional
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
        if cytemplate is None:
            cytemplate = CyTemplate.fromTemplate(
                template, 
                simplify=self.simplify,
            )

        _derivs = zeros((5, template.x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _derivs,
                flux,
                fwhm,
                split,
                left,
                right,
                cytemplate,
                scale,
                n_scales,
            )

        if interpolation_matrix is None:
            _derivs = _interp2d(x, template.x, _derivs)
        else:
            _derivs = _interp2d_matrix(_derivs, interpolation_matrix)

        return _derivs if derivs is None else add(derivs, _derivs, out=derivs)
