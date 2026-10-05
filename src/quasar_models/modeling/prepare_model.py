from collections.abc import Iterable
from dataclasses import field
from typing import Protocol, Self, TypedDict, TypeVar

from numpy import allclose, diff, interp, median
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import CompoundModel_
from quasar_typing.numpy import BoolVector, FloatMatrix, FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.resolving import mask_kernels

from .base_model import BaseModel
from .sequential_model import SequentialModel

M = TypeVar("M", BaseModel, CompoundModel_[BaseModel])


def _is_logbinned(x: FloatVector) -> bool:
    """
    Return True if the input array `x` is log-binned, i.e. if the relative 
    differences between consecutive wavelength values are approximately 
    constant.
    """
    dv = diff(x) / x[:-1]
    dv_median = median(dv)
    return allclose(dv, dv_median)


class _CoordsProtocol(Protocol):
    _x: FloatVector
    primary_mask: BoolVector
    secondary_mask: BoolVector
    _kernels: FloatMatrix
    _original: bool


class _DataDict(TypedDict):
    x: FloatVector
    y: FloatVector
    dy: FloatVector

@dataclass
class PrepareModel:
    # Rest wavelength grid
    x: FloatVector = field(kw_only=True)
    # The model to prepare
    model: M = field(kw_only=True)

    # Mask: where to evaluate the model
    eval_mask: BoolVector = field(kw_only=True)
    # Mask: where to compare the model and data
    comp_mask: BoolVector = field(kw_only=True)

    # Resolution kernels for instrumental broadening
    kernels: FloatMatrix | csr_matrix_ | None = field(default=None, kw_only=True)

    # Whether the wavelength array is log-binned
    is_logbinned: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.x.shape != self.eval_mask.shape:
            raise ValueError(
                f"Shape mismatch between `x` ({self.x.shape}) and "
                f"`eval_mask` ({self.eval_mask.shape})."
            )
        if self.x.shape != self.comp_mask.shape:
            raise ValueError(
                f"Shape mismatch between `x`/`eval_mask` ({self.x.shape}) and "
                f"`comp_mask` ({self.comp_mask.shape})."
            )
        if self.kernels is not None \
            and self.kernels.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"Dimension 0 mismatch between `x` ({self.x.shape}) and "
                f"`kernels` ({self.kernels.shape})."
            )
        
        if self.is_logbinned is None:
            self.is_logbinned = _is_logbinned(self.x)

    @classmethod
    def fromCoords(
        cls, 
        coords: _CoordsProtocol, 
        model: M,
    ) -> Self:
        return cls(
            x=coords._x,
            model=model,
            eval_mask=coords.primary_mask,
            comp_mask=coords.secondary_mask,
            kernels=coords._kernels,
            is_logbinned=not coords._original,
        )

    def createDataDict(self, coords: _CoordsProtocol) -> _DataDict:
        return {
            "x": coords._x[self.eval_mask],
            "y": coords._y[self.eval_mask],
            "dy": coords._dy[self.eval_mask],
        }

    @classmethod
    def prepare_submodels(
        cls,
        x: FloatVector,
        submodels: Iterable[BaseModel],
        is_logbinned: bool | None = None,
        fixeds: dict[str, dict[str, bool]] | None = None,
    ) -> None:
        """Prepare each submodel. 
        
        If a submodel is a subclass of `TemplateModel`, 
        it is prepared with the full wavelength array to construct interpolation 
        matrices. 
        
        If a submodel is a `GaussianModel` or a subclass of 
        `_VProfileCopy`, and the wavelength array is NOT log-binned, the 
        submodel's `dx` property is set.

        Parameters
        ----------
        x : FloatVector
            The wavelength array the models will be evaluated on.
        submodels : Iterable[BaseModel]
            The submodels to be prepared.
        is_logbinned : bool | None, optional
            Whether the wavelength array is log-binned. If None, it will be 
            inferred.
        fixeds : dict[str, dict[str, bool]] | None, optional
            A dictionary designating which parameters are fixed. This parameter
            should be used with caution. 
        """
        _dx = diff(x)
        if is_logbinned is None:
            is_logbinned = _is_logbinned(x)

        for submodel in submodels:
            fixed = None
            if (fixeds is not None) and submodel.name in fixeds:
                fixed = fixeds[submodel.name]

            if submodel.model_type in {"fe", "ba", "hg"}:
                submodel.prepare_model(x, fixed=fixed)
            elif submodel.model_type == "em":
                if is_logbinned:
                    dx = None
                else:
                    dx = interp(
                        submodel.wave, x[:-1], _dx, 
                        left=_dx[0], 
                        right=_dx[-1],
                    )
                submodel.prepare_model(dx, fixed=fixed)
            else:
                submodel.prepare_model(fixed=fixed)

    @classmethod
    def prepare_model(
        cls,
        x: FloatVector,
        model: M,
        is_logbinned: bool | None = None,
        fixeds: dict[str, dict[str, bool]] | None = None,
    ) -> None:
        submodels = (model,) if model.n_submodels == 1 else model
        cls.prepare_submodels(
            x, 
            submodels, 
            is_logbinned=is_logbinned, 
            fixeds=fixeds,
        )

    @classmethod
    def unprepare_submodels(
        cls,
        submodels: Iterable[BaseModel],
    ) -> None:
        for submodel in submodels:
            submodel.unprepare_model()

    @classmethod
    def unprepare_model(
        cls,
        model: M,
    ) -> None:
        submodels = (model,) if model.n_submodels == 1 else model
        cls.unprepare_submodels(submodels)

    def __enter__(self) -> tuple[SequentialModel, BoolVector | None]:
        """
        Enter the context manager, preparing the model for evaluation:
        1. Determine the evaluation and comparison grids based on the masks.
        2. Prepare the sequential model for evaluation. 
        3. Return the prepared sequential model and, if applicable, the 
        comparison mask for the evaluation grid.
        """
        if self.kernels is None or self.kernels.shape[0] == 1:
            # Instrumental broadening has no effect 
            # -> evaluate and compare on the same grid
            # -> set `kernels` to None
            # -> set `where` to None
            x_eval = self.x[self.eval_mask]
            kernels = None
            where = None
        else:
            # Instrumental broadening has an effect
            # -> evaluate on the evaluation grid
            # -> mask the resolution kernels accordingly
            # -> compare on the comparison grid
            x_eval = self.x[self.eval_mask]
            kernels = mask_kernels(
                self.kernels,
                self.eval_mask,
            )
            where = self.comp_mask[self.eval_mask]

        seq_model = SequentialModel(self.model, kernels=kernels)
        self.prepare_model(
            x_eval, 
            self.model, 
            is_logbinned=self.is_logbinned,
            fixeds=seq_model.get_true_fixed(),
        )
        return seq_model, where

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.unprepare_model(self.model)
