__all__ = ['BalmerContinuumTemplate']

from typing import Self, ClassVar
from numpy import interp, searchsorted
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath
from quasar_typing.numpy import SortedFloatVector

from quasar_utils.setup import Info

from .evaluation import evaluate
from .io import PATH_TO_CACHE, load, save, save_to_cache, load_from_cache

from ...utils.template import BaseTemplate

@dataclass(eq=False)
class BalmerContinuumTemplate(BaseTemplate):
    temp: float = field(kw_only=True)
    tau: float = field(kw_only=True)
    scale: float = field(kw_only=True)
    boltz: float = field(kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()
        _ = AbsoluteDirPath._validate(self.PATH_TO_CACHE)

        if self.normalisation is None:
            idx = searchsorted(self.fwhm, self.fwhm_norm, side='right') - 1
            self.normalisation = interp(self.x_norm, self.x, self.data[idx])

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) \
            and (self.temp == other.temp) \
            and (self.tau == other.tau) \
            and (self.scale == other.scale) \
            and (self.boltz == other.boltz)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update({
            'temp': self.temp,
            'tau': self.tau,
            'scale': self.scale,
            'boltz': self.boltz,
        })
        return state

    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, '_alpha_matrix', None) is not None

        return BalmerContinuumTemplate(
            fwhm=self.fwhm.copy(), 
            x=self.x.copy(), 
            data=self.data.copy(),
            is_logspace=self.is_logspace,
            sigma_res=self.sigma_res,
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
        info: Info,
        is_logspace: bool = False,
        name: str = "no_name",
    ) -> Self:
        sigma_res = info.loading.sigma_res
        edge = info.balmer.edge
        boltz = info.units.getBoltzmannFactor()

        _data = evaluate(
            x, 1.0, fwhm[0], temp, tau, scale, 
            sigma_res=sigma_res, edge=edge, boltz=boltz, normalisation=1.0,
        )
        obj = BalmerContinuumTemplate(
            fwhm=fwhm[:1],
            x=x,
            data=_data[None,:],
            is_logspace=is_logspace,
            sigma_res=sigma_res,
            name=name,
            path=None,
            _alpha_matrix=None,
            _beta_matrix=None,
            _xn=None,
            temp=temp,
            tau=tau,
            scale=scale,
            x_norm=edge,
            fwhm_norm=info.balmer.fwhm_norm,
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
