__all__ = ["BalmerContinuumTemplate"]

from dataclasses import field
from typing import ClassVar, Self

from pydantic.dataclasses import dataclass
from quasar_typing.numpy import SortedFloatVector
from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath
from quasar_utils.setup import Info

from quasar_models._core.modeling.balmer.continuum import evaluate
from quasar_models.modeling.template import BaseTemplate

from .io import PATH_TO_CACHE, load, load_from_cache, save, save_to_cache


@dataclass(eq=False)
class BalmerContinuumTemplate(BaseTemplate):
    temp: float = field(kw_only=True)
    tau: float = field(kw_only=True)
    scale: float = field(kw_only=True)
    boltz: float = field(kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.normalisation is None:
            self._calculate_normalisation()

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and (self.temp == other.temp)
            and (self.tau == other.tau)
            and (self.scale == other.scale)
            and (self.boltz == other.boltz)
        )

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update(
            {
                "temp": self.temp,
                "tau": self.tau,
                "scale": self.scale,
                "boltz": self.boltz,
            }
        )
        return state

    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, "_alpha_matrix", None) is not None

        return BalmerContinuumTemplate(
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
            temp=self.temp,
            tau=self.tau,
            scale=self.scale,
            x_norm=self.x_norm,
            fwhm_norm=self.fwhm_norm,
            normalisation=self.normalisation,
            boltz=self.boltz,
        )

    @classmethod
    def instantiate(
        cls,
        fwhm: SortedFloatVector,
        x: SortedFloatVector,
        temp: float,
        tau: float,
        scale: float,
        *,
        sigma_res: float,
        n_scales: float,
        edge: float,
        fwhm_norm: float,
        boltz: float,
        is_logspace: bool = False,
        name: str = "no_name",
    ) -> Self:
        _data = evaluate(
            x,
            1.0,
            fwhm[0],
            temp=temp,
            tau=tau,
            scale=scale,
            edge=edge,
            boltz=boltz,
            sigma_res=sigma_res,
            n_scales=n_scales,
            normalisation=1.0,
        )[None, :]
        obj = BalmerContinuumTemplate(
            fwhm=fwhm[:1],
            x=x,
            data=_data,
            is_logspace=is_logspace,
            sigma_res=sigma_res,
            n_scales=n_scales,
            name=name,
            path=None,
            _alpha_matrix=None,
            _beta_matrix=None,
            _xn=None,
            temp=temp,
            tau=tau,
            scale=scale,
            x_norm=edge,
            fwhm_norm=fwhm_norm,
            normalisation=None,
            boltz=boltz,
        )
        return obj.upsample(fwhm, inplace=True, keep_x=True)

    # I/O

    def save(
        self,
        *,
        path: str | AbsoluteFITSPath,
        info: Info,
    ) -> AbsoluteFITSPath:
        return save(
            template=self,
            path=path,
            info=info,
        )

    @classmethod
    def load(
        cls,
        *,
        path: str | AbsoluteFITSPath,
        info: Info,
    ) -> Self:
        kwargs = load(path=path, info=info)
        return BalmerContinuumTemplate(**kwargs)

    # I/O from '.cache' directory

    def save_to_cache(self, info: Info) -> AbsoluteFITSPath:
        return save_to_cache(template=self, info=info)

    @classmethod
    def load_from_cache(
        cls,
        *,
        temp: float,
        tau: float,
        scale: float,
        info: Info,
    ) -> Self:
        kwargs = load_from_cache(
            temp=temp,
            tau=tau,
            scale=scale,
            info=info,
        )
        return BalmerContinuumTemplate(**kwargs)
