__all__ = ['BalmerContinuumTemplate']

from typing import Self, ClassVar
from numpy import interp
from functools import lru_cache
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath
from quasar_typing.numpy import SortedFloatVector

from quasar_utils.setup import Info

from .evaluation import evaluate
from .io import PATH_TO_CACHE, load, save, save_to_cache, load_from_cache

from ...utils.template import BaseTemplate

@dataclass
class BalmerContinuumTemplate(BaseTemplate):
    temp: float
    tau: float
    scale: float

    x_norm: float = field(init=False)
    fwhm_norm: float | None = field(default=None, kw_only=True)
    normalisation: float | None = field(default=None, kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()
        _ = AbsoluteDirPath._validate(self.PATH_TO_CACHE)

        self.x_norm = self.info.balmer.edge
        self.fwhm_norm = self.fwhm_norm or self.info.host.fwhm_norm
        if self.normalisation is None:
            self.normalisation = interp(
                self.x_norm,
                self.x,
                evaluate(
                    self.x, 
                    1.0, self.fwhm_norm, self.temp, self.tau, self.scale, 
                    sigma_res=self.info.loading.sigma_res,
                    edge=self.x_norm,
                    boltz=self.info.units.getBoltzmannFactor(),
                    normalisation=1.0,
                ),
            )

    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, '_alpha_matrix', None) is not None

        return BalmerContinuumTemplate(
            self.fwhm.copy(), 
            self.x.copy(), 
            self.data.copy(),
            self.temp,
            self.tau,
            self.scale,
            info=self.info,
            is_logspace=self.is_logspace,
            name=self.name,
            path=self.path,
            _alpha_matrix=self._alpha_matrix if with_matrices else None,
            _beta_matrix=self._beta_matrix if with_matrices else None,
            _xn=self._xn if with_matrices else None,
            fwhm_norm=self.fwhm_norm,
            normalisation=self.normalisation,
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
        _data = evaluate(
            x, 1.0, fwhm[0], temp, tau, scale, 
            sigma_res=info.loading.sigma_res,
            edge=info.balmer.edge,
            boltz=info.units.getBoltzmannFactor(),
        )
        obj = BalmerContinuumTemplate(
            fwhm[:1],
            x,
            _data[None,:],
            temp, tau, scale,
            info=info,
            is_logspace=is_logspace,
            name=name,
            path=None,
        )
        return obj.upsample(fwhm, inplace=True, keep_x=True)
    
    # I/O
        
    def save(self, path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
        return save(self, path)

    @classmethod
    @lru_cache(maxsize=None)
    def load(cls, path: str | AbsoluteFITSPath, info: Info) -> Self:
        args, kwargs = load(path, info)
        return BalmerContinuumTemplate(*args, **kwargs)

    # I/O from '.cache' directory

    def save_to_cache(self) -> AbsoluteFITSPath:
        return save_to_cache(self)

    @classmethod
    def load_from_cache(
        cls, 
        temp: float,
        tau: float,
        scale: float,
        *,
        info: Info,
    ) -> Self:
        args, kwargs = load_from_cache(temp, tau, scale, info=info)
        return BalmerContinuumTemplate(*args, **kwargs)
