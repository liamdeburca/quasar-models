__all__ = ['BalmerSeriesTemplate']

from typing import Self, ClassVar, Literal
from numpy import empty, float64, log, searchsorted, array
from dataclasses import field
from pydantic.dataclasses import dataclass
from functools import lru_cache

from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath
from quasar_typing.numpy import SortedFloatVector, FloatVector

from quasar_utils.setup import Info

from .evaluation import evaluate
from .io import PATH_TO_CACHE, load, save, save_to_cache, load_from_cache

from ...utils.template import BaseTemplate

@dataclass
class BalmerSeriesTemplate(BaseTemplate):
    waves: SortedFloatVector
    weights: FloatVector
    temp: float
    dens: float
    n_u_range: tuple[int, int]
    
    # Only series data from Storey&Hummer1995 is currently supported
    name: Literal['sh1995'] = field(default='sh1995', kw_only=True)

    x_norm: float = field(init=False)
    fwhm_norm: float | None = field(default=None, kw_only=True)
    normalisation: float | None = field(default=None, kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()

        _ = AbsoluteDirPath._validate(self.PATH_TO_CACHE)

        if self.waves.size != self.weights.size:
            msg = "Sizes of 'waves' ({}) and 'weights' ({}) must match.".format(
                self.waves.size, self.weights.size,
            )
            raise ValueError(msg)
                
        self.x_norm = self.info.balmer.edge

        if self.fwhm_norm is None:
            self.fwhm_norm = self.info.balmer.fwhm_norm

        if self.normalisation is None:
            self.normalisation = evaluate(
                self.x_norm,
                1.0,
                self.fwhm_norm,
                sigma_res=self.info.loading.sigma_res,
                edge=self.x_norm,
                waves=self.waves,
                weights=self.weights,
                normalisation=1.0,
            )

    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, '_alpha_matrix', None) is not None

        return BalmerSeriesTemplate(
            self.fwhm.copy(), 
            self.x.copy(), 
            self.data.copy(),
            self.waves.copy(),
            self.weights.copy(),
            self.temp,
            self.dens,
            self.n_u_range,
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

    def crop(self, n_u_range: tuple[int, int]) -> Self:
        """
        Creates a new BalmerSeriesTemplate with a subset of the upper level 
        ('n_u') range.
        """
        n_u_lower = n_u_range[0] or 2
        n_u_upper = n_u_range[1] or 100

        n_us_current = range(max(self.n_u_range), min(self.n_u_range) - 1, -1)

        n_us: list[int] = []
        waves: list[float] = []
        weights: list[float] = []
        for n_u, wave, weight in zip(n_us_current, self.waves, self.weights):
            if (n_u < n_u_lower) or (n_u > n_u_upper):
                continue

            n_us.append(n_u)
            waves.append(wave)
            weights.append(weight)

        return self.instantiate(
            self.fwhm.copy(),
            self.x.copy(),
            array(waves, dtype=float64),
            array(weights, dtype=float64) / sum(weights),
            self.temp,
            self.dens,
            (min(n_us), max(n_us)),
            info=self.info,
            fwhm_norm=self.fwhm_norm,
            is_logspace=self.is_logspace,
            name=self.name,
        )
    
    @classmethod
    def instantiate(
        cls,
        fwhm: SortedFloatVector,
        x: SortedFloatVector,
        waves: FloatVector,
        weights: FloatVector,
        temp: float,
        dens: float,
        n_u_range: tuple[int, int],
        *,
        info: Info,
        fwhm_norm: float | None = None,
        normalisation: float | None = None,
        is_logspace: bool = False,
        name: Literal['sh1995'] = 'sh1995',
    ) -> Self:
        _data = evaluate(
            x, 1.0, fwhm[0],
            sigma_res=info.loading.sigma_res,
            edge=info.balmer.edge,
            waves=waves,
            weights=weights,
            normalisation=1.0,
        )
        obj = BalmerSeriesTemplate(
            fwhm[:1],
            x,
            _data[None,:],
            waves,
            weights,
            temp, dens, n_u_range,
            fwhm_norm=fwhm_norm,
            normalisation=normalisation,
            info=info,
            is_logspace=is_logspace,
            name=name,
            path=None,
        )
        return obj.upsample(fwhm, inplace=True)
    
    def upsample(
        self,
        fwhm: SortedFloatVector,
        inplace: bool = False,
    ) -> Self:
        """
        Upsamples the BalmerSeriesTemplate to the specified FWHM values.
        """        
        obj = self if inplace else self.copy(with_matrices=True)

        log_x = log(self.x)
        log_edge = log(self.x_norm)
        log_waves = log(self.waves)

        data = empty(shape=(fwhm.size, self.x.size), dtype=float64)
        indices = searchsorted(self.fwhm, fwhm)

        for i, fwhm_curr in enumerate(fwhm):
            if indices[i] < self.fwhm.size \
                and self.fwhm[indices[i]] == fwhm_curr:
                data[i,:] = self.data[indices[i],:]
            else:
                data[i,:] = evaluate(
                    self.x,
                    flux=1.0,
                    fwhm=fwhm_curr,
                    sigma_res=self.info.loading.sigma_res,
                    edge=self.x_norm,
                    waves=self.waves,
                    weights=self.weights,
                    log_x=log_x,
                    log_edge=log_edge,
                    log_waves=log_waves,
                    normalisation=1.0,
                )

        obj.fwhm = fwhm
        obj.data = data

        return obj
    
    # I/O
    
    def save(self, path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
        return save(self, path)

    @classmethod
    @lru_cache(maxsize=None)
    def load(cls, path: str | AbsoluteFITSPath, info: Info) -> Self:
        args, kwargs = load(path, info)
        return BalmerSeriesTemplate(*args, **kwargs)

    # I/O from '.cache' directory

    def save_to_cache(self) -> AbsoluteFITSPath:
        return save_to_cache(self)

    @classmethod
    def load_from_cache(
        cls, 
        name: Literal['sh1995'],
        temp: float,
        dens: float,
        n_u_range: tuple[int, int],
        *,
        info: Info,
    ) -> Self:
        args, kwargs = load_from_cache(name, temp, dens, n_u_range, info=info)
        obj = BalmerSeriesTemplate(*args, **kwargs)
        return obj if obj.n_u_range == n_u_range else obj.crop(n_u_range)