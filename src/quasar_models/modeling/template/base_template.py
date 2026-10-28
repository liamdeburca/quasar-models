__all__ = ["BaseTemplate"]

import math
from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from quasar_typing.bounds import AstropyBounds
from quasar_typing.literals import TemplateName
from quasar_typing.numpy import FloatMatrix, FloatVector, SortedFloatVector
from quasar_typing.pathlib import DirPath, FITSPath
from quasar_typing.scipy import CSRMatrix
from quasar_utils.setup import Info

from quasar_models._core.convolution import convolve_signal, kernel

templates_dir: Path = Path(__file__).parent / "templates"


def drop_nan(arr: FloatVector) -> FloatVector:
    return arr[np.isfinite(arr)]


def drop_nonpos(arr: FloatVector) -> FloatVector:
    return arr[np.isfinite(arr) & (arr > 0)]


SIGMA_TO_FWHM: float = 2 * (2 * math.log(2)) ** 0.5
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM

TemplateTuple = tuple[FloatMatrix, FloatVector, FloatVector]


@dataclass
class BaseTemplate(ABC):
    fwhm: SortedFloatVector = field(kw_only=True)
    x: SortedFloatVector = field(kw_only=True)
    data: FloatMatrix = field(kw_only=True)

    is_logspace: bool = field(kw_only=True)
    sigma_res: float | None = field(default=None, kw_only=True)
    n_scales: float = field(default=5.0, kw_only=True)
    name: TemplateName = field(kw_only=True)

    path: FITSPath | None = field(default=None, kw_only=True)

    _alpha_matrix: CSRMatrix | None = field(
        default=None, repr=False, kw_only=True
    )
    _beta_matrix: CSRMatrix | None = field(
        default=None, repr=False, kw_only=True
    )
    _xn: SortedFloatVector | None = field(
        default=None, repr=False, kw_only=True
    )

    x_norm: float = field(kw_only=True)
    fwhm_norm: float = field(kw_only=True)
    normalisation: float | None = field(default=None, kw_only=True)

    PATH_TO_CACHE: ClassVar[DirPath]

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.fwhm = np.ascontiguousarray(self.fwhm, dtype=np.float64)
        self.data = np.ascontiguousarray(self.data, dtype=np.float64)

        if self.is_logspace:
            assert self.sigma_res is not None
        else:
            assert self.sigma_res is None

        self._validate_shapes()

    def __getstate__(self) -> dict:
        # Serialisation is primarily used in fitting routines, where binning
        # matrices are not needed.
        return {
            "fwhm": self.fwhm,
            "x": self.x,
            "data": self.data,
            "is_logspace": self.is_logspace,
            "sigma_res": self.sigma_res,
            "name": self.name,
            "path": self.path,
            "x_norm": self.x_norm,
            "fwhm_norm": self.fwhm_norm,
            "normalisation": self.normalisation,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state)

    def _validate_shapes(self):
        if self.fwhm.size != self.data.shape[0]:
            msg = f"fwhm size {self.fwhm.size} does not match first axis of data {self.data.shape[0]}"
            raise ValueError(msg)
        if self.x.size != self.data.shape[1]:
            msg = f"x size {self.x.size} does not match second axis of data {self.data.shape[1]}"
            raise ValueError(msg)

    def __getitem__(self, *sel) -> Self:
        """
        Create a copy of the Template based on the selection.
        """
        if len(sel) > 2:
            raise IndexError("Too many indices for Template.")

        if len(sel) == 1:
            # 'x' doesn't change -> can keep transformation matrices
            obj = self.copy(with_matrices=True)
            obj.data = obj.data[sel[0], :]
            obj.fwhm = obj.fwhm[sel[0]]
        else:
            # 'x' changes -> cannot keep transformation matrices
            obj = self.copy(with_matrices=False)
            obj.data = obj.data[sel[0], :][:, sel[1]]
            obj.fwhm = obj.fwhm[sel[0]]
            obj.x = obj.x[sel[1]]

        return obj

    def __eq__(self, other: object) -> bool:
        """
        True if all attributes are equal.
        """
        return (
            isinstance(other, self.__class__)
            and (self.name == other.name)
            and (self.is_logspace == other.is_logspace)
            and (self.sigma_res == other.sigma_res)
            and (self.path == other.path)
            and (self.x_norm == other.x_norm)
            and (self.fwhm_norm == other.fwhm_norm)
            and (self.normalisation == other.normalisation)
            and np.array_equal(self.fwhm, other.fwhm)
            and np.array_equal(self.x, other.x)
            and np.array_equal(self.data, other.data)
        )

    def _calculate_normalisation(self) -> float:
        if (self.x_norm < self.x[0]) or (self.x[-1] < self.x_norm):
            # Edge case where the normalisation cannot be computed
            # accurately.
            self.normalisation = 1.0
        else:
            idx = np.searchsorted(self.fwhm, self.fwhm_norm, side="right") - 1
            self.normalisation = np.interp(self.x_norm, self.x, self.data[idx])

        return self.normalisation

    def normalise(self, inplace: bool = False) -> Self:
        obj = self if inplace else self.copy(with_matrices=True)

        if (
            hasattr(self, "normalisation")
            and self.normalisation is not None
            and self.normalisation != 0
        ):
            obj.data /= self.normalisation
            obj.normalisation = 1.0

        return obj

    @abstractmethod
    def copy(self, with_matrices: bool = False) -> Self:
        """
        Creates a copy of the current Template. If `with_matrices` is True, the
        logspace-transformation matrices are also copied, if available.
        """

    def upsample(
        self,
        fwhm: SortedFloatVector,
        inplace: bool = False,
        keep_x: bool = False,
        *,
        sigma_res: float | None = None,
    ) -> Self:
        """
        Upsamples the Template to the specified FWHM values.
        """
        if self.is_logspace:
            obj = self if inplace else self.copy(with_matrices=True)

            data = np.empty(shape=(fwhm.size, self.x.size), dtype=np.float64)
            indices = np.searchsorted(self.fwhm, fwhm)

            fwhm_prev = self.fwhm[0]
            data_prev = self.data[0, :]

            for i, fwhm_curr in enumerate(fwhm):
                # Check if exact match exists at the insertion index
                if (
                    indices[i] < self.fwhm.size
                    and self.fwhm[indices[i]] == fwhm_curr
                ):
                    data[i, :] = self.data[indices[i]]
                else:
                    k = kernel(
                        (fwhm_curr**2 - fwhm_prev**2) ** 0.5,
                        self.sigma_res,
                        self.n_scales,
                    )
                    data[i, :] = convolve_signal(data_prev, k)

                fwhm_prev = fwhm_curr
                data_prev = data[i, :]

            obj.fwhm = fwhm
            obj.data = data
        else:
            assert sigma_res is not None

            template = self.createLogspace(
                sigma_res=sigma_res,
                keep_x=keep_x,
            )
            template.upsample(fwhm, inplace=True)
            obj = self.mimicLogspace(template, inplace=inplace)

        return obj

    def resample(
        self,
        fwhm: SortedFloatVector,
        keep_x: bool = False,
    ) -> Self:
        """
        Resamples the Template to the specified FWHM values.
        """
        return self[:1].upsample(fwhm, inplace=True, keep_x=keep_x)

    ### I/O: Abstract methods ###

    @abstractmethod
    def save(
        self,
        path: FITSPath,
        overwrite: bool = False,
    ) -> FITSPath:
        """
        Saves the Template to a FITS file.
        """

    @classmethod
    @abstractmethod
    def load(
        cls,
        *,
        path: str | FITSPath,
        info: Info,
    ) -> Self:
        pass

    ### Space transformations ###

    @validate_call
    def createLogspace(
        self,
        *,
        sigma_res: float,
        xr: SortedFloatVector | None = None,
        keep_x: bool = False,
        conserve: bool = True,
    ) -> Self:
        """
        Creates a logspace equivalent of the current Template.

        Parameters
        ----------
        xr : FloatVector | None
            The new logspace coordinates. If None, they will be generated
            based on `sigma_res`.
        inplace : bool, optional
            Whether to modify the current Template or return a new one.
            Default is False.

        Returns
        -------
        template : Template
            The logspace-equivalent template.
        """
        from quasar_utils.binning import _alpha_matrix_sparse, edges, gradient

        if self.is_logspace:
            dx = self.x * self.sigma_res
            x_edges = edges(self.x, self.x * self.sigma_res)
        else:
            dx = gradient(self.x)
            x_edges = edges(self.x)

        if not (dx > 0).all():
            raise ValueError("All elements of dx must be positive.")

        if xr is None:
            nr = math.log(x_edges[-1] / x_edges[0]) // math.log(1 + sigma_res) + 1
            logxr_edges = math.log(x_edges[0]) + sigma_res * np.arange(nr + 1)
            xr_edges = np.exp(logxr_edges)
            xr = np.exp(0.5 * (logxr_edges[:-1] + logxr_edges[1:]))
        else:
            logxr = np.log(xr)
            dlogxr = np.full_like(xr, fill_value=sigma_res)
            logxr_edges = np.empty(xr.size + 1, dtype=float)
            logxr_edges[:-1] = logxr - dlogxr / 2
            logxr_edges[-1] = logxr[-1] + dlogxr[-1] / 2
            xr_edges = np.exp(logxr_edges)

        if self.is_logspace:
            return self.interpolate(xr)

        dxr = xr * sigma_res
        if keep_x:
            xn_edges = x_edges
        else:
            dxn = np.median(dx)
            n_left = int(abs(x_edges[0] - xr_edges[0]) // dxn + 1)
            n_right = int(abs(xr_edges[-1] - x_edges[-1]) // dxn + 1)

            xn_edges = np.empty(int(n_left + len(x_edges) + n_right))
            xn_edges[:n_left] = x_edges[0] + dxn * np.arange(-n_left, 0)
            xn_edges[n_left:-n_right] = x_edges
            xn_edges[-n_right:] = x_edges[-1] + dxn * np.arange(1, n_right + 1)


        cp = self.copy()
        cp.is_logspace = True
        cp.sigma_res = sigma_res
        cp._xn = 0.5 * (xn_edges[:-1] + xn_edges[1:])
        cp.x = xr

        self._alpha_matrix = cp._alpha_matrix = (
            _alpha_matrix_sparse(
                x_edges,
                xr_edges,
                dx=dx,
                dxr=dxr,
                conserve=conserve,
            )
        )
        self._beta_matrix = cp._beta_matrix = _alpha_matrix_sparse(
            xr_edges,
            xn_edges,
            dx=dxr,
            dxr=dx if keep_x else np.diff(xn_edges),
            conserve=conserve,
        )
        cp.data = np.maximum(cp._alpha_matrix.dot(cp.data.T).T, 0)

        # cp.normalisation = None
        cp.__post_init__()

        return cp

    def mimicLogspace(
        self,
        template: Self,
        inplace: bool = False,
    ) -> Self:
        """
        Mimics the logspace-equivalent of this template, i.e. the two templates
        must share identical beta matrices (logspace -> any space).

        This method updates the fwhm and data arrays using the
        logspace-equivalent template, and is therefore useful for when the
        logspace-equivalent templates has been upsampled/resampled.

        Parameters
        ----------
        template : Template
            The logspace-equivalent template to mimic.
        inplace : bool, optional
            Whether to modify the current template or a copy.
            Default is False.
        """
        from numpy import maximum

        assert template.is_logspace, "Provided template must be in logspace."
        assert not self.is_logspace, "Current template must be in linspace."

        assert (
            self._beta_matrix is not None
        ), "Current template must have a beta matrix."
        assert (
            template._beta_matrix is not None
        ), "Provided template must have a beta matrix."
        assert (
            self._beta_matrix is template._beta_matrix
        ), "Templates must share the same beta matrix if 'inplace=True'."

        if inplace:
            obj = self
        else:
            obj = self.copy(with_matrices=True)

        obj.fwhm = template.fwhm.copy()
        obj.x = template._xn.copy()
        obj.data = maximum(self._beta_matrix.dot(template.data.T).T, 0)

        # Delete transformation matrices as x and data may have been transformed
        # differently
        del obj._alpha_matrix
        del obj._beta_matrix
        del obj._xn

        return obj

    @validate_call
    def interpolate(self, x: SortedFloatVector) -> Self:
        """
        Interpolates the template to match the new x coordinates.

        If the new x coordinates match the current x coordinates, the same
        template is returned (or a copy if `inplace=False`).
        """
        from quasar_utils.interpolation import _create_interp_matrix
        obj = self.copy(with_matrices=False)
        if np.array_equal(x, self.x):
            return obj

        M, b = _create_interp_matrix(self.x, x, left=0, right=0)
        obj.data = np.maximum(np.array([M.dot(y) + b for y in self.data]), 0)
        obj.x = x

        return obj

    @validate_call
    def rasterise(
        self,
        x: SortedFloatVector,
        y: FloatVector,
        dy: FloatVector,
        *,
        flux_bounds: AstropyBounds = (None, None),
        fwhm_bounds: AstropyBounds = (None, None),
    ) -> tuple[FloatVector, FloatVector]:
        """
        Performs a raster fit of the template to the provided data returning the
        chi-square and flux value for each FWHM.
        """
        from quasar_utils.raster import _rasterise
        obj = self.interpolate(x)
        return _rasterise(
            y,
            dy,
            obj.fwhm,
            obj.data,
            flux_bounds=flux_bounds,
            fwhm_bounds=fwhm_bounds,
        )

    def simplifyData(
        self,
        tol: float = 1e-8,
        inplace: bool = False,
    ) -> Self:
        assert tol > 0.0
        obj = self if inplace else self.copy(with_matrices=True)

        for i, arr in enumerate(obj.data):
            mask = arr < tol * arr.max()
            obj.data[i, mask] = 0.0

        return obj

    def create1DTemplate(
        self,
        fwhm: float,
    ) -> Self:
        """
        Creates a template with a single FWHM value. This is useful for
        template-based models with fixed FWHM values.
        """
        assert fwhm >= self.fwhm[0]

        idx = np.searchsorted(self.fwhm, fwhm, side="right") - 1
        if idx == -1:
            raise ValueError(
                "FWHM is smaller than the smallest available FWHM in the "
                f"template: {fwhm:.1f} < {self.fwhm[0]:.1f}"
            )

        if fwhm == self.fwhm[idx]:
            return self[idx : idx + 1]

        # Create new column and return only the newly created column
        return self.resample(np.array([self.fwhm[0], fwhm]), keep_x=True)[1:2]
    