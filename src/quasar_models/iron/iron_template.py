__all__ = ["IronTemplate"]

from dataclasses import field
from typing import Any, ClassVar, Literal, Self

from numpy import searchsorted
from pydantic.dataclasses import dataclass
from quasar_typing.numpy import FloatVector
from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath
from quasar_utils.setup import Info

from quasar_models._core.modeling.split import evaluate as split_evaluate
from quasar_models.modeling.template import BaseTemplate

from .io import PATH_TO_CACHE, load, load_from_cache, save, save_to_cache


@dataclass(eq=False)
class IronTemplate(BaseTemplate):
    """
    Template class specifically designed for Iron pseudo-continua.
    """

    name: Literal["vw2001", "v2003", "bw"] | str = field(default="vw2001", kw_only=True)

    x_norm: float | None = field(default=None, kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.x_norm is None) or (self.normalisation is None):
            self._calculate_normalisation()

    def _calculate_normalisation(self) -> float:
        idx = searchsorted(self.fwhm, self.fwhm_norm, side="right") - 1
        y = self.data[idx]
        self.x_norm = self.x[y.argmax()]
        self.normalisation = y.max()
        return self.normalisation

    def copy(self, with_matrices: bool = False) -> Self:
        """
        Creates a copy of the current IronTemplate instance. If `with_matrices`
        is True, the logspace-transformation matrices are also copied, if
        available.
        """
        with_matrices &= getattr(self, "_alpha_matrix", None) is not None
        return IronTemplate(
            fwhm=self.fwhm.copy(),
            x=self.x.copy(),
            data=self.data.copy(),
            is_logspace=self.is_logspace,
            sigma_res=self.sigma_res,
            n_scales=self.n_scales,
            name=self.name,
            path=self.path,
            _alpha_matrix=self._alpha_matrix if with_matrices else None,
            _beta_matrix=self._beta_matrix if with_matrices else None,
            _xn=self._xn if with_matrices else None,
            x_norm=self.x_norm,
            fwhm_norm=self.fwhm_norm,
            normalisation=self.normalisation,
        )

    def applySplit(
        self,
        split: float,
        left: float,
        right: float,
        scale: float,
        inplace: bool = False,
    ) -> Self:
        assert self.is_logspace, "Template must be in logspace."

        obj = self if inplace else self.copy(with_matrices=True)

        obj.data *= self._get_split_weight(
            obj.x,
            split,
            left,
            right,
            scale,
        )[None, :]

        return obj

    def _get_split_weight(
        self,
        x: FloatVector,
        split: float,
        left: float,
        right: float,
        scale: float,
    ) -> FloatVector:
        """
        Calculates the split weight vector for the given parameters.

        Notes
        -----
        This method assumes that the x-array is in logspace (logbinned).
        """
        return split_evaluate(
            x,
            split,
            left,
            right,
            sigma_res=self.sigma_res,
            scale=scale,
        )

    def save(
        self,
        *,
        path: str | AbsoluteFITSPath,
        info: Info,
    ) -> AbsoluteFITSPath:
        """
        Saves the IronTemplate to a FITS file.
        """
        return save(template=self, path=path, info=info)

    @classmethod
    def load(
        cls,
        *,
        path: str | AbsoluteFITSPath,
        info: Info,
    ) -> Self:
        kwargs = load(path=path, info=info)
        return IronTemplate(**kwargs)

    def save_to_cache(self, info: Info) -> AbsoluteFITSPath:
        return save_to_cache(template=self, info=info)

    @classmethod
    def load_from_cache(
        cls,
        *,
        name: Literal["vw2001", "v2003", "bw"] | str,
        info: Info,
    ) -> Self:
        kwargs = load_from_cache(name=name, info=info)
        return IronTemplate(**kwargs)

    ### Serialization

    @classmethod
    def deserialize(cls, data: dict[str, Any], info: Info) -> Self:
        template = cls.load_from_cache(name=data["name"], info=info)
        template = cls._deserialize_helper(template, data, info)
        
        template.fwhm_norm = info.iron.fwhm_norm
        template.normalisation = None
        template.__post_init__()
        return template