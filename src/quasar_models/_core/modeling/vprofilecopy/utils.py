from collections.abc import Callable

from numpy import float64, zeros
from quasar_typing.numpy import FloatMatrix, FloatVector

from . import evaluate, fit_deriv


class _VProfileCopyBase:
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable:
        raise NotImplementedError

    def __init__(
        self,
        func_name: str | None = None,
    ) -> None:
        self.func_name: str | None = func_name
        self.__wrapped__: Callable | None = self._get_wrapped(func_name)

    def __getstate__(self) -> dict:
        return {
            "func_name": self.func_name,
        }

    def __setstate__(self, state: dict) -> None:
        self.func_name = state["func_name"]
        self.__wrapped__ = self._get_wrapped(self.func_name)


class VProfileCopyEvaluateV(_VProfileCopyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strengths: FloatVector,
        fwhm_vs: FloatVector,
        v_offs: FloatVector,
        *,
        wave: float,
        sigma_res: float,
        y: FloatVector | None = None,
    ) -> FloatVector:
        """
        Evaluate a velocity profile copy on the given wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        strength_scale : float
            Scaling factor for the strengths of the Gaussian components.
        strengths : FloatVector
            Integrated fluxs of the Gaussian components.
        fwhm_vs : FloatVector
            Intrinsic FWHM (km/s) of the Gaussian components.
        v_offs : FloatVector
            Velocity offsets (km/s) of the Gaussian components.
        wave : float
            Theoretical wavelength of the emission line.
        sigma_res : float
            Velocity resolution of the spectrum (c).
        y : FloatVector, optional
            Output array to store the evaluated velocity profile copy. If not 
            provided, a new array will be created.

        Returns
        -------
        y : FloatVector
            Evaluated velocity profile copy.
        """
        if y is None:
            y = zeros(x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                y, x, strength_scale, strengths, fwhm_vs, v_offs, wave, sigma_res
            )
        return y


class VProfileCopyFitDerivV(_VProfileCopyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strengths: FloatVector,
        fwhm_vs: FloatVector,
        v_offs: FloatVector,
        *,
        wave: float,
        sigma_res: float,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        """
        Calculate the partial derivatives of a velocity-profile copy.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        strength_scale : float
            Scaling factor for the strengths of the Gaussian components.
        strengths : FloatVector
            Integrated fluxs of the Gaussian components.
        fwhm_vs : FloatVector
            Intrinsic FWHM (km/s) of the Gaussian components.
        v_offs : FloatVector
            Velocity offsets (km/s) of the Gaussian components.
        wave : float
            Theoretical wavelength of the emission line.
        sigma_res : float
            Velocity resolution of the spectrum (c).
        derivs : FloatMatrix, optional
            Output array to store the calculated derivatives. If not provided, a
            new array will be created.

        Returns
        -------
        derivs : FloatMatrix
            Calculated partial derivatives of the velocity profile copy.
        """
        if derivs is None:
            derivs = zeros((1 + 3 * strengths.size, x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                derivs, x, strength_scale, strengths, fwhm_vs, v_offs, wave, sigma_res
            )
        return derivs

###

class VProfileCopyEvaluateX(_VProfileCopyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(evaluate, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strengths: FloatVector,
        fwhm_vs: FloatVector,
        v_offs: FloatVector,
        *,
        wave: float,
        dx: float,
        y: FloatVector | None = None,
    ) -> FloatVector:
        """
        Evaluate a velocity profile copy on the given wavelength array.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        strength_scale : float
            Scaling factor for the strengths of the Gaussian components.
        strengths : FloatVector
            Integrated fluxs of the Gaussian components.
        fwhm_vs : FloatVector
            Intrinsic FWHM (km/s) of the Gaussian components.
        v_offs : FloatVector
            Velocity offsets (km/s) of the Gaussian components.
        wave : float
            Theoretical wavelength of the emission line.
        dx : float
            Wavelength resolution of the spectrum.
        y : FloatVector, optional
            Output array to store the evaluated velocity profile copy. If not 
            provided, a new array will be created.

        Returns
        -------
        y : FloatVector
            Evaluated velocity profile copy.
        """
        if y is None:
            y = zeros(x.size, dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                y, x, strength_scale, strengths, fwhm_vs, v_offs, wave, dx,
            )
        return y

    
class VProfileCopyFitDerivX(_VProfileCopyBase):
    @classmethod
    def _get_wrapped(cls, func_name: str | None) -> Callable | None:
        return getattr(fit_deriv, func_name) if func_name else None

    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strengths: FloatVector,
        fwhm_vs: FloatVector,
        v_offs: FloatVector,
        *,
        wave: float,
        dx: float,
        derivs: FloatMatrix | None = None,
    ) -> FloatMatrix:
        """
        Calculate the partial derivatives of a velocity-profile copy.

        Parameters
        ----------
        x : FloatVector
            Wavelength array.
        strength_scale : float
            Scaling factor for the strengths of the Gaussian components.
        strengths : FloatVector
            Integrated fluxs of the Gaussian components.
        fwhm_vs : FloatVector
            Intrinsic FWHM (km/s) of the Gaussian components.
        v_offs : FloatVector
            Velocity offsets (km/s) of the Gaussian components.
        wave : float
            Theoretical wavelength of the emission line.
        dx : float
            Wavelength resolution of the spectrum.
        derivs : FloatMatrix, optional
            Output array to store the calculated derivatives. If not provided, a
            new array will be created.

        Returns
        -------
        derivs : FloatMatrix
            Calculated partial derivatives of the velocity profile copy.
        """
        if derivs is None:
            derivs = zeros((1 + 3 * strengths.size, x.size), dtype=float64)
        if self.__wrapped__ is not None:
            self.__wrapped__(
                derivs, x, strength_scale, strengths, fwhm_vs, v_offs, wave, dx
            )
        return derivs
