from collections.abc import Callable

from numpy import add, float64, zeros
from quasar_typing.numpy import FloatMatrix, FloatVector

from ..template.cytemplate import CyTemplate
from ..utils import _interp, _interp2d, _interp2d_matrix, _interp_matrix
from . import evaluate, fit_deriv


class _HostGalaxyBase:
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


class HostGalaxyEvaluate(_HostGalaxyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        *,
        template: object,
        cytemplate: CyTemplate | None = None,
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
        template : object
            Host galaxy template object.
        cytemplate : CyTemplate, optional
            Compiled Cython template object. If not provided, it will be created 
            from the template
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
                cytemplate,
                n_scales,
            )

        if interpolation_matrix is None:
            _y = _interp(x, template.x, _y)
        else:
            _y = _interp_matrix(_y, interpolation_matrix)

        return _y if y is None else add(y, _y, out=y)


class HostGalaxyFitDeriv(_HostGalaxyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        *,
        cytemplate: CyTemplate | None = None,
        template: object,
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
        cytemplate : CyTemplate, optional
            Compiled Cython template object. If not provided, it will be created 
            from the template.
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
        if cytemplate is None:
            cytemplate = CyTemplate.fromTemplate(
                template,
                simplify=self.simplify,
            )

        _derivs = zeros((2, template.x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                _derivs,
                flux,
                fwhm,
                cytemplate,
                n_scales,
            )

        if interpolation_matrix is None:
            _derivs = _interp2d(x, template.x, _derivs)
        else:
            _derivs = _interp2d_matrix(_derivs, interpolation_matrix)

        return _derivs if derivs is None else add(derivs, _derivs, out=derivs)
