__all__ = ['BalmerSeriesTemplate']

from typing import Self, ClassVar, Literal
from numpy import empty, float64, searchsorted, array, array_equal, interp
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AbsoluteDirPath, AbsoluteFITSPath
from quasar_typing.numpy import SortedFloatVector, FloatVector

from quasar_utils.setup import Info

from quasar_core.modelling.balmer.series import evaluate
from .io import PATH_TO_CACHE, load, save, save_to_cache, load_from_cache

from ...utils.template import BaseTemplate

@dataclass(eq=False)
class BalmerSeriesTemplate(BaseTemplate):
    waves: SortedFloatVector = field(kw_only=True)
    weights: FloatVector = field(kw_only=True)
    temp: float = field(kw_only=True)
    dens: float = field(kw_only=True)
    n_u_range: tuple[int, int] = field(kw_only=True)
    
    # Only series data from Storey&Hummer1995 is currently supported
    name: Literal['sh1995'] = field(default='sh1995', kw_only=True)

    PATH_TO_CACHE: ClassVar[AbsoluteDirPath] = PATH_TO_CACHE

    def __post_init__(self) -> None:
        super().__post_init__()
        self.weights /= self.weights.sum()

        _ = AbsoluteDirPath._validate(self.PATH_TO_CACHE)

        if self.waves.size != self.weights.size:
            msg = "Sizes of 'waves' ({}) and 'weights' ({}) must match.".format(
                self.waves.size, self.weights.size,
            )
            raise ValueError(msg)
                
        if self.normalisation is None:
            idx = searchsorted(self.fwhm, self.fwhm_norm, side='right') - 1
            self.normalisation = interp(self.x_norm, self.x, self.data[idx])

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) \
            and array_equal(self.waves, other.waves) \
            and array_equal(self.weights, other.weights) \
            and (self.temp == other.temp) \
            and (self.dens == other.dens) \
            and (self.n_u_range == other.n_u_range)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update({
            'waves': self.waves,
            'weights': self.weights,
            'temp': self.temp,
            'dens': self.dens,
            'n_u_range': self.n_u_range,
        })
        return state

    def copy(self, with_matrices: bool = False) -> Self:
        with_matrices &= getattr(self, '_alpha_matrix', None) is not None

        return BalmerSeriesTemplate(
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
            waves=self.waves.copy(),
            weights=self.weights.copy(),
            temp=self.temp,
            dens=self.dens,
            n_u_range=self.n_u_range,
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

        waves = array(waves, dtype=float64)
        weights = array(weights, dtype=float64)

        return self.instantiate(
            self.fwhm.copy(),
            self.x.copy(),
            waves,
            weights / weights.sum(),
            self.temp,
            self.dens,
            (min(n_us), max(n_us)),
            sigma_res=self.sigma_res,
            n_scales=self.n_scales,
            edge=self.x_norm,
            fwhm_norm=self.fwhm_norm,
            normalisation=None,
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
        sigma_res: float,
        n_scales: float,
        edge: float,
        fwhm_norm: float,
        normalisation: float | None = None,
        is_logspace: bool = False,
        name: Literal['sh1995'] = 'sh1995',
    ) -> Self:
        weights /= weights.sum()

        _data = evaluate(
            x, 1.0, fwhm[0],
            sigma_res=sigma_res,
            waves=waves,
            weights=weights,
            edge=edge,
            normalisation=1.0,
        )[None,:]
        obj = BalmerSeriesTemplate(
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
            x_norm=edge,
            fwhm_norm=fwhm_norm,
            normalisation=normalisation,
            waves=waves,
            weights=weights,
            temp=temp,
            dens=dens,
            n_u_range=n_u_range,
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
        
        data = empty(shape=(fwhm.size, self.x.size), dtype=float64)
        indices = searchsorted(self.fwhm, fwhm)

        for i, fwhm_curr in enumerate(fwhm):
            if indices[i] < self.fwhm.size \
                and self.fwhm[indices[i]] == fwhm_curr:
                data[i,:] = self.data[indices[i],:]
            else:
                evaluate(
                    self.x,
                    flux=1.0,
                    fwhm=fwhm_curr,
                    sigma_res=self.sigma_res,
                    waves=self.waves,
                    weights=self.weights,
                    edge=self.x_norm,
                    normalisation=1.0,
                    y=data[i,:],
                )

        obj.fwhm = fwhm
        obj.data = data

        return obj
    
    # I/O
    
    def save(
        self, 
        *,
        path: str | AbsoluteFITSPath,
        info: Info,
    ) -> AbsoluteFITSPath:
        """
        Saves the BalmerSeriesTemplate to a FITS file.
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
        return BalmerSeriesTemplate(**kwargs)

    # I/O from '.cache' directory

    def save_to_cache(self, info: Info) -> AbsoluteFITSPath:
        return save_to_cache(template=self, info=info)

    @classmethod
    def load_from_cache(
        cls, 
        *,
        name: Literal['sh1995'],
        temp: float,
        dens: float,
        n_u_range: tuple[int, int],
        info: Info,
    ) -> Self:
        kwargs = load_from_cache(
            name=name, 
            temp=temp, 
            dens=dens, 
            n_u_range=n_u_range, 
            info=info,
        )
        obj = BalmerSeriesTemplate(**kwargs)
        return obj if obj.n_u_range == n_u_range else obj.crop(n_u_range)