from typing import Self, ClassVar, Literal
from numpy import interp
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath

from quasar_utils.setup import Info
from quasar_utils.decorators import lru_cache

from .io import PATH_TO_CACHE, save, load, save_to_cache, load_from_cache
from ..utils.template import BaseTemplate, evaluate as template_evaluate

@dataclass
class HostGalaxyTemplate(BaseTemplate):
    """
    Template class specifically designed for Host Galaxy templates.
    """
    name: Literal['bc2003'] = field(default='bc2003', kw_only=True)
    age: int = field(default=0, kw_only=True)

    x_norm: float | None = field(default=None, kw_only=True)
    fwhm_norm: float | None = field(default=None, kw_only=True)
    normalisation: float | None = field(default=None, kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()

        _ = AbsoluteDirPath._validate(self.PATH_TO_CACHE)

        self.x_norm = self.x_norm or self.info.host.x_norm
        self.fwhm_norm = self.fwhm_norm or self.info.host.fwhm_norm

        if self.normalisation is None:
            self.normalisation = interp(
                self.x_norm, 
                self.x, 
                template_evaluate(
                    self.x, 1.0, self.fwhm_norm, 
                    template=self, normalisation=1.0,
                ),
            )

    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, '_alpha_matrix', None) is not None

        return HostGalaxyTemplate(
            self.fwhm.copy(), 
            self.x.copy(), 
            self.data.copy(),
            age=self.age,
            x_norm=self.x_norm,
            fwhm_norm=self.fwhm_norm,
            normalisation=self.normalisation,
            info=self.info,
            is_logspace=self.is_logspace,
            name=self.name,
            path=self.path,
            _alpha_matrix=self._alpha_matrix if with_matrices else None,
            _beta_matrix=self._beta_matrix if with_matrices else None,
            _xn=self._xn if with_matrices else None,
        )        

    # I/O
    
    def save(self, path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
        return save(self, path)

    @classmethod
    @lru_cache(maxsize=None)
    def load(cls, path: str | AbsoluteFITSPath, info: Info) -> Self:
        args, kwargs = load(path, info)
        return HostGalaxyTemplate(*args, **kwargs)

    # I/O from '.cache' directory

    def save_to_cache(self) -> AbsoluteFITSPath:
        return save_to_cache(self)

    @classmethod
    def load_from_cache(cls, name: str, age: int, *, info: Info) -> Self:
        args, kwargs = load_from_cache(name, age, info=info)
        return HostGalaxyTemplate(*args, **kwargs)
