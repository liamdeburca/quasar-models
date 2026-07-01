from typing import Self, ClassVar, Literal
from numpy import interp, searchsorted
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath

from quasar_utils.setup import Info

from .io import PATH_TO_CACHE, save, load, save_to_cache, load_from_cache
from ..utils.template import BaseTemplate

@dataclass(eq=False)
class HostGalaxyTemplate(BaseTemplate):
    """
    Template class specifically designed for Host Galaxy templates.
    """
    name: Literal['bc2003'] = field(default='bc2003', kw_only=True)
    age: int = field(default=0, kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()

        _ = AbsoluteDirPath._validate(self.PATH_TO_CACHE)

        if self.normalisation is None:
            idx = searchsorted(self.fwhm, self.fwhm_norm, side='right') - 1
            self.normalisation = interp(self.x_norm, self.x, self.data[idx])

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) \
            and self.age == other.age

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['age'] = self.age
        return state
    
    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, '_alpha_matrix', None) is not None
        return HostGalaxyTemplate(
            fwhm=self.fwhm.copy(), 
            x=self.x.copy(), 
            data=self.data.copy(),
            is_logspace=self.is_logspace,
            sigma_res=self.sigma_res,
            n_scales=self.n_scales,
            name=self.name,
            path=self.path,
            _alpha_matrix=self._alpha_matrix.copy() if with_matrices else None,
            _beta_matrix=self._beta_matrix.copy() if with_matrices else None,
            _xn=self._xn.copy() if with_matrices else None,
            x_norm=self.x_norm,
            fwhm_norm=self.fwhm_norm,
            normalisation=self.normalisation,
            age=self.age,
        )

    # I/O
    
    def save(
        self, 
        *,
        path: str | AbsoluteFITSPath,
        info: Info,
    ) -> AbsoluteFITSPath:
        return save(template=self, path=path, info=info)

    @classmethod
    def load(
        cls, 
        *,
        path: str | AbsoluteFITSPath, 
        info: Info,
    ) -> Self:
        kwargs = load(path=path, info=info)
        return HostGalaxyTemplate(**kwargs)

    # I/O from '.cache' directory

    def save_to_cache(self, info: Info) -> AbsoluteFITSPath:
        return save_to_cache(template=self, info=info)

    @classmethod
    def load_from_cache(
        cls, 
        *,
        name: str, 
        age: int,
        info: Info,
    ) -> Self:
        kwargs = load_from_cache(name=name, age=age, info=info)
        return HostGalaxyTemplate(**kwargs)
