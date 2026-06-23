__all__ = [
    '_VProfileCopy',
]

from typing import Self, Callable, Iterable, ClassVar, Literal
from itertools import product
from numpy import stack, zeros, float64

from quasar_models.line.gaussian import GaussianModel
from quasar_models.utils.basemodel import BaseModel
from quasar_models.utils.astropy import apply_bounds
from quasar_models.tying import IdenticalTie

from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import CompoundModel_ 

from .. import evaluation

def _evaluate(
    x: float | FloatVector,
    strength_scale: float,
    strength: tuple[float],
    sigma_v: tuple[float],
    v_off: tuple[float],
    *,
    wave: float = None,
    sigma_res: float = None,
) -> float | FloatVector:
    return strength_scale * sum(
        evaluation.evaluate(x, *args, wave, sigma_res) \
        for args in zip(strength, sigma_v, v_off)
    )

def _fit_deriv(
    x: FloatVector,
    strength_scale: float,
    strength: tuple[float],
    sigma_v: tuple[float],
    v_off: tuple[float],
    *,
    wave: float = None,
    sigma_res: float = None,
    n_profiles: int = None,
    fixed: dict[str, bool] | None = None,
) -> list[FloatVector]:
    
    dfs = zeros((1 + 3 * n_profiles, len(x)), dtype=float64)
        
    if fixed is None:
        fixed = {'strength_scale': False}
        fixed.update({f'strength_{i}': False for i in range(1, n_profiles+1)})
        fixed.update({f'sigma_v_{i}': False for i in range(1, n_profiles+1)})
        fixed.update({f'v_off_{i}': False for i in range(1, n_profiles+1)})
        
    if not all(fixed.values()):
        if not fixed['strength_scale']:
            dfs[0,:] = _evaluate(
                x, 
                1.0, 
                strength, sigma_v, v_off,
                wave=wave, sigma_res=sigma_res,
            )

        for i, args in enumerate(zip(strength, sigma_v, v_off)):
            _fixed: dict[str, bool] = {
                'strength': fixed[f"strength_{i+1}"],
                'sigma_v':  fixed[f"sigma_v_{i+1}"],
                'v_off':    fixed[f"v_off_{i+1}"],
            }
            if all(_fixed.values()): 
                continue

            dfs[3*i+1:3*i+4,:] = strength_scale * evaluation.fit_deriv(
                x,
                *args,
                wave=wave, sigma_res=sigma_res, 
                fixed=_fixed,
            )

    return list(dfs)

###

class _VProfileCopy(BaseModel):
    n_profiles: ClassVar[Literal[1, 2, 3, 4, 5]]
    model_type: ClassVar[Literal['em']] = 'em'

    @property
    def pure_name(self) -> str:
        return self.name

    @property
    def wave(self) -> float: 
        return self.meta['wave']

    @wave.setter
    def wave(self, value: float) -> None:
        self.meta['wave'] = value

    @property
    def sigma_res(self) -> float | None: 
        return self.meta['sigma_res']
    
    @sigma_res.setter
    def sigma_res(self, value: float | None) -> None:
        self.meta['sigma_res'] = value

    @property
    def master_name(self) -> str | None: 
        return self.meta['master_name']

    @master_name.setter
    def master_name(self, value: str | None) -> None:
        self.meta['master_name'] = value

    @property
    def n_sigmas(self) -> float | None: 
        return self.meta['n_sigmas']
    
    @n_sigmas.setter
    def n_sigmas(self, value: float | None) -> None:
        self.meta['n_sigmas'] = value

    @classmethod
    def create(
        cls,
        wave: float,
        name: str,
        *gs: GaussianModel,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,

        sigma_res: float | None = None,
        master_name: float | None = None,
        n_sigmas: float | None = None,

        adapt: bool = True,
        freeze: bool = False,
    ) -> Self:
        meta = cls._get_metadata(
            wave,
            name,
            *gs,
            sigma_res=sigma_res,
            master_name=master_name,
            n_sigmas=n_sigmas,
        )
        model = cls(
            strength_scale_value,
            name=name,
            meta=meta,
        )
        model.strength_scale.value = strength_scale_value
        model.strength_scale.bounds = strength_scale_bounds
        model.strength_scale.fixed = strength_scale_fixed

        if adapt:
            model._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        if freeze:
            model._freeze_velocity_profile(inplace=True)
            model._forget_ties(inplace=True)

        return model

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: GaussianModel | Iterable[GaussianModel],
        *,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,

        adapt: bool = True,
        freeze: bool = False,
    ) -> Self:
        if cls.n_profiles == 1:
            assert isinstance(model, GaussianModel)
            model = (model,)
        if cls.n_profiles > 1:
            model = tuple(model)
            assert len(model) == cls.n_profiles

        return cls.create(
            wave,
            name,
            *model,
            strength_scale_value=strength_scale_value,
            strength_scale_bounds=strength_scale_bounds,
            strength_scale_fixed=strength_scale_fixed,
            adapt=adapt,
            freeze=freeze,
        )
    
    def makeCopy(
        self,
        master_model: GaussianModel | Iterable[GaussianModel],
        *,
        strength_scale_value: float = 1.0,
        strength_scale_bounds: tuple[float | None, float | None] = (0, None),
        strength_scale_fixed: bool = False,
        freeze: bool = False,
    ) -> Self:
        return self.from_model(
            self.wave,
            self.name,
            master_model,
            strength_scale_value=strength_scale_value,
            strength_scale_bounds=strength_scale_bounds,
            strength_scale_fixed=strength_scale_fixed,
            adapt=True,
            freeze=freeze,
        )

    @classmethod
    def _get_metadata(
        cls,
        wave: float,
        name: str,
        *gs: GaussianModel,
        sigma_res: float | None = None,
        master_name: float | None = None,
        n_sigmas: float | None = None,
    ) -> dict:
        meta = {
            'pure_name': name,
            'wave': wave,
        }

        if len(gs) == 0:
            # Used when copying
            meta['sigma_res'] = sigma_res
            meta['master_name'] = master_name
            meta['n_sigmas'] = n_sigmas
        elif len(gs) == cls.n_profiles:
            meta['sigma_res'] = gs[0].sigma_res
            meta['master_name'] = gs[0].pure_name
            meta['n_sigmas'] = gs[0].n_sigmas

        return meta

    def __str__(self) -> str:
        name = self.name
        master = self.master_name
        wave = self.wave
        untied = hasattr(self, '_prev_ties')
        return f"{self.__class__.__name__}({name=}, {master=}, {wave=}, untied={untied})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def evaluate(
        self,
        x, 
        strength_scale,
        *args,
    ):
        strengths = tuple(args[3*i] for i in range(self.n_profiles))
        sigma_vs  = tuple(args[3*i + 1] for i in range(self.n_profiles))
        v_offs    = tuple(args[3*i + 2] for i in range(self.n_profiles))
        return _evaluate(
            x,
            strength_scale,
            strengths,
            sigma_vs,
            v_offs,
            wave=self.wave,
            sigma_res=self.sigma_res,
        )
    
    def fit_deriv(self, x, strength_scale, *args):
        strengths = tuple(args[3*i] for i in range(self.n_profiles))
        sigma_vs  = tuple(args[3*i + 1] for i in range(self.n_profiles))
        v_offs    = tuple(args[3*i + 2] for i in range(self.n_profiles))
        return _fit_deriv(
            x,
            strength_scale,
            strengths,
            sigma_vs,
            v_offs,
            wave=self.wave,
            sigma_res=self.sigma_res,
            n_profiles=self.n_profiles,
            fixed = self.fixed,
        )

    def jac(self, x, strength_scale, *args):
        return stack(self.fit_deriv(x, strength_scale, *args), axis=0)
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        return (4.0, self.wave)
    
    ### Utilities

    def _adapt_to_models(
        self,
        *gs: GaussianModel,
        tie_vel_profile: bool = True,
        inplace: bool = False,
    ) -> Self:
        """
        Lorem ipsum.

        Parameters
        ----------
        *gs : GaussianModel
        tie_vel_profile : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        for (i, g), pname in product(
            enumerate(gs, start=1), 
            ('strength', 'sigma_v', 'v_off'),
        ):
            attr_name: str = f"{pname}_{i}"
            attr = getattr(model, attr_name)
            attr.value = getattr(g, pname).value
            attr.bounds = getattr(g, pname).bounds

            if tie_vel_profile: 
                attr.tied = IdenticalTie.from_model(g, pname)
            
        return model
    
    def _freeze_velocity_profile(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Fixes all profile parameters: 'strength', 'sigma_v', and 'v_off'.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies this instance. If False, returns a modified copy.
            Defaults to False.

        Returns
        -------
        _VProfileCopy
            Modified instance (self if inplace=True, otherwise a copy).

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        for i, pname in product(
            range(1, model.n_profiles+1), 
            ('strength', 'sigma_v', 'v_off'),
        ):
            getattr(model, f"{pname}_{i}").fixed = True

        return model
    
    def _thaw_velocity_profile(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Unfixes all profile parameters: 'strength', 'sigma_v', and 'v_off'.

        Parameters
        ----------
        inplace : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        for i, pname in product(
            range(1, model.n_profiles+1), 
            ('strength', 'sigma_v', 'v_off'),
        ):
            getattr(model, f"{pname}_{i}").fixed = False

        return model
    
    def _forget_ties(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Removes all profile ties ('strength', 'sigma_v', and 'v_off') and 
        stores them in the '_prev_ties' attribute. 

        Parameters
        ----------
        inplace : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()

        if not hasattr(model, '_prev_ties'):
            _prev_ties: dict[str, Callable] = {}
            
            for i, pname in product(
                range(1, model.n_profiles+1), 
                ('strength', 'sigma_v', 'v_off'),
            ):
                attr_name = f"{pname}_{i}"
                attr = getattr(model, attr_name)
                if tie := attr.tied:
                    _prev_ties[attr_name] = tie
                    attr.tied = False

            setattr(model, '_prev_ties', _prev_ties)

        return model
    
    def _remember_ties(
        self,
        inplace: bool = False,
    ) -> Self:
        """
        Restores all profile ties ('strength', 'sigma_v', and 'v_off') from the 
        '_prev_ties' attribute and deletes '_prev_ties'.

        Parameters
        ----------
        inplace : bool, optional

        Returns
        -------
        Self

        Notes
        -----
        Lorem ipsum.
        """
        model = self if inplace else self.copy()   
        
        if hasattr(model, '_prev_ties'):
            for attr_name, tie in model._prev_ties.items():
                getattr(model, attr_name).tied = tie

            del model._prev_ties

        return model

    def adaptStrengthScale(
        self,
        model: CompoundModel_[GaussianModel] | Iterable[GaussianModel],
    ) -> None:
        """
        Lorem ipsum.

        Parameters
        ----------
        model : CompoundModel_[GaussianModel] or Iterable[GaussianModel]

        Notes
        -----
        Lorem ipsum.
        """
        current_strength = sum(
            getattr(self, f"strength_{i}").value \
            for i in range(1, self.n_profiles+1)
        )
        model_strength = sum(
            m.strength.value 
            for m in (model if isinstance(model, Iterable) else [model])
        )
        self.strength_scale.value = apply_bounds(
            model_strength / current_strength, 
            self.strength_scale.bounds,
        )

    def splitIntoGaussians(self) -> list[GaussianModel]:
        """
        Lorem ipsum.

        Returns
        -------
        list[GaussianModel]

        Notes
        -----
        Lorem ipsum.
        """
        gaussian_models: list[GaussianModel] = []

        names: list[str] = (
            [self.pure_name] \
            if self.n_profiles == 1
            else [self.pure_name + f"#{1+i}" for i in range(self.n_profiles)]
        )
        for i, name in enumerate(names):
            g = GaussianModel(self.wave, self.sigma_res, name=name)

            for attr_name in ('strength', 'sigma_v', 'v_off'):
                getattr(g, attr_name).value = getattr(self, f"{attr_name}_{i+1}").value
                getattr(g, attr_name).bounds = getattr(self, f"{attr_name}_{i+1}").bounds

            gaussian_models.append(g)

        return gaussian_models