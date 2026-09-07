from collections.abc import Callable

from numpy import add, float64, zeros
from quasar_typing.numpy import FloatMatrix, FloatVector

from ..template.cytemplate import CyTemplate
from ..utils import _interp, _interp2d, _interp2d_matrix, _interp_matrix
from . import evaluate, fit_deriv


class _BalmerBase:
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable:
        raise NotImplementedError

    def __init__(
        self,
        func_name: str | None = None,
        simplify: bool = False,
    ) -> None:
        self.func_name: str | None = func_name
        self.simplify: bool = simplify
        self.__wrapped__: Callable | None = self._get_wrapped(func_name)

    def __getstate__(self) -> dict:
        return {
            "func_name": self.func_name,
            "simplify": self.simplify,
        }

    def __setstate__(self, state: dict) -> None:
        self.func_name = state["func_name"]
        self.simplify = state["simplify"]
        self.__wrapped__ = self._get_wrapped(self.func_name)


###


class BalmerEvaluate(_BalmerBase):
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
        continuum_template: object,
        series_template: object,
        continuum_cytemplate: CyTemplate | None = None,
        series_cytemplate: CyTemplate | None = None,
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
         continuum_template : object
            Continuum template object.
        series_template : object
            Series template object.
        continuum_cytemplate : CyTemplate, optional
        series_cytemplate : CyTemplate, optional
        n_scales : float
        interpolation_matrix : tuple, optional
        y : FloatVector, optional
            Existing flux density array to modify in place.

        Returns
        -------
        y : FloatVector
            Flux density array.
        """
        if continuum_cytemplate is None:
            continuum_cytemplate = CyTemplate.fromTemplate(
                continuum_template,
                simplify=self.simplify,
            )
        if series_cytemplate is None:
            series_cytemplate = CyTemplate.fromTemplate(
                series_template,
                simplify=self.simplify,
            )

        _y = zeros(continuum_template.x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _y,
                flux,
                fwhm,
                ratio,
                continuum_cytemplate,
                series_cytemplate,
                n_scales,
            )
        if interpolation_matrix is None:
            _y = _interp(x, continuum_template.x, _y)
        else:
            _y = _interp_matrix(_y, interpolation_matrix)

        return _y if y is None else add(y, _y, out=y)


class BalmerFitDeriv(_BalmerBase):
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
        continuum_template: object,
        series_template: object,
        continuum_cytemplate: CyTemplate | None = None,
        series_cytemplate: CyTemplate | None = None,
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
         continuum_template : object
            Continuum template object.
        series_template : object
            Series template object.
        continuum_cytemplate : CyTemplate, optional
        series_cytemplate : CyTemplate, optional
        n_scales : float
        interpolation_matrix : tuple, optional
        derivs : FloatMatrix, optional
            Existing partial derivatives array to modify in place.

        Returns
        -------
        derivs : FloatMatrix
            Partial derivatives array.
        """
        if continuum_cytemplate is None:
            continuum_cytemplate = CyTemplate.fromTemplate(continuum_template)
        if series_cytemplate is None:
            series_cytemplate = CyTemplate.fromTemplate(series_template)

        _derivs = zeros((3, continuum_template.x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _derivs,
                flux,
                fwhm,
                ratio,
                continuum_cytemplate,
                series_cytemplate,
                n_scales,
            )
        if interpolation_matrix is None:
            _derivs = _interp2d(x, continuum_template.x, _derivs)
        else:
            _derivs = _interp2d_matrix(_derivs, interpolation_matrix)

        return _derivs if derivs is None else add(derivs, _derivs, out=derivs)
