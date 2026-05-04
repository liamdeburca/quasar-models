"""
    Lorem ipsum.
"""
__all__ = [
    'VProfileCopy1G',
    'VProfileCopy2G',
    'VProfileCopy3G',
    'VProfileCopy4G',
    'VProfileCopy5G',
]
from abc import abstractmethod
from typing import Self, Callable, Iterable, ClassVar
from astropy.modeling import Parameter
from functools import partial
from itertools import product
from numpy import zeros_like, float64
from numpy.typing import NDArray

from . import evaluation
from .gaussian import GaussianModel
from ..utils.basemodel import BaseModel
from ..utils.astropy import apply_bounds

from quasar_typing.astropy import CompoundModel_ 

def _evaluate(
    x: float | NDArray[float64],
    strength_scale: float,
    strength: tuple[float],
    sigma_v: tuple[float],
    v_off: tuple[float],
    *,
    wave: float = None,
    sigma_res: float = None,
) -> float | NDArray[float64]:
    return strength_scale * sum(
        evaluation.evaluate(x, *args, wave, sigma_res) \
        for args in zip(strength, sigma_v, v_off)
    )

def _fit_deriv(
    x: NDArray[float64],
    strength_scale: float,
    strength: tuple[float],
    sigma_v: tuple[float],
    v_off: tuple[float],
    *,
    wave: float = None,
    sigma_res: float = None,
    n_profiles: int = None,
    fixed: dict[str, bool] = None,
) -> list[NDArray[float64]]:
    
    dfs: list[NDArray[float64]] = []
    dfs.extend(zeros_like(x, dtype=float64) for _ in range(1 + 3 * n_profiles))

    if fixed is None:
        fixed = {'strength_scale': False}
        for i in range(1, n_profiles+1):
            fixed[f'strength_{i}'] = False
            fixed[f'sigma_v_{i}']  = False
            fixed[f'v_off_{i}']    = False

    elif not all(fixed.values()):
        if not fixed['strength_scale']:
            dfs[0][:] = _evaluate(
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

            _dfs = evaluation.fit_deriv(
                x,
                *args,
                wave=wave, sigma_res=sigma_res, 
                fixed=_fixed,
            )
            dfs[3*i + 1][:] = strength_scale * _dfs[0]
            dfs[3*i + 2][:] = strength_scale * _dfs[1]
            dfs[3*i + 3][:] = strength_scale * _dfs[2]

    return dfs

def tie_parameters(
    compound_model, 
    model_name: str, 
    param_name: str,
) -> float:
    return getattr(compound_model[model_name], param_name).value    

###

class _VProfileCopy(BaseModel):
    def __str__(self) -> str:
        name = self.name
        master = self.master_name
        wave = self.wave
        untied = hasattr(self, '_prev_ties')
        return f"{self.__class__.__name__}({name=}, {master=}, {wave=}, untied={untied})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    @abstractmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: GaussianModel | Iterable[GaussianModel],
        freeze: bool = False,
    ) -> Self:
        pass
    
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
    
    def fit_deriv(
        self,
        x, 
        strength_scale,
        *args,
    ):
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
    
    @property
    def model_type(self) -> str:
        """
        Lorem ipsum.

        Returns
        -------
        str

        Notes
        -----
        Lorem ipsum.
        """
        return 'em'
    
    @property
    def sorting_key(self) -> tuple[float, float]:
        """
        Lorem ipsum.

        Returns
        -------
        tuple[float, float]

        Notes
        -----
        Lorem ipsum.
        """
        return (
            2.0,        # 0: power law, 1: iron emission, 2: line emission
            self.wave   # sort from bluest to reddest centre. 
        )
    
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
                attr.tied = partial(
                    tie_parameters,
                    model_name = g.name,
                    param_name = pname,
                )
            
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
        current_strength = sum([
            getattr(self, f"strength_{i}").value \
            for i in range(1, self.n_profiles+1)
        ])
        model_strength = sum([
            m.strength.value 
            for m in (model if isinstance(model, Iterable) else [model])
        ])
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
            g.strength = getattr(self, f"strength_{i+1}")
            g.sigma_v = getattr(self, f"sigma_v_{i+1}")
            g.v_off = getattr(self, f"v_off_{i+1}")

            gaussian_models.append(g)

        return gaussian_models

class VProfileCopy1G(_VProfileCopy):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength_scale : Parameter
        Multiplicative scaling factor for the profile strength. Bounds (0, None).
    strength_1 : Parameter
        Strength of the Gaussian profile. Fixed by default.
    sigma_v_1 : Parameter
        Velocity dispersion in $c$ units. Fixed by default.
    v_off_1 : Parameter
        Velocity offset relative to rest wavelength. Fixed by default.
    wave : float
        Central wavelength of the profile.
    sigma_res : float
        Instrumental resolution from the source GaussianModel.
    pure_name : str
        Base name without submodel identifiers.
    master_name : str
        Name of the source Gaussian model.
    """
    n_profiles: ClassVar[int] = 1

    strength_scale = Parameter(default=1, bounds=(0, None))
    
    strength_1 = Parameter(default=1, fixed=True)
    sigma_v_1  = Parameter(default=1e-3, fixed=True)
    v_off_1    = Parameter(default=0, fixed=True)

    def __init__(
        self,
        wave: float,
        name: str,
        *gs: GaussianModel,
        **kwargs,
    ):
        assert len(gs) == self.n_profiles

        super().__init__(name=name, **kwargs)

        self.pure_name:   str = name
        self.wave:      float = wave
        self.sigma_res: float = gs[0].sigma_res
        self.master_name: str = gs[0].pure_name

        self._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        self._freeze_velocity_profile(inplace=True)
        self._forget_ties(inplace=True)

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: GaussianModel | Iterable[GaussianModel],
        freeze: bool = False,
        **kwargs,
    ) -> Self:
        if isinstance(model, GaussianModel):
            model = (model,)
        else:
            model = tuple(model)
            assert len(model) == 1

        vprof = VProfileCopy1G(wave, name, *model, **kwargs)
        if freeze:
            vprof._freeze_velocity_profile(inplace=True)
            vprof._forget_ties(inplace=True)

        return vprof
    
class VProfileCopy2G(_VProfileCopy):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength_scale : Parameter
        Multiplicative scaling factor for both profile strengths. Bounds (0, None).
    strength_1, strength_2 : Parameter
        Strengths of the Gaussian profiles. Fixed by default.
    sigma_v_1, sigma_v_2 : Parameter
        Velocity dispersions in $c$ units. Fixed by default.
    v_off_1, v_off_2 : Parameter
        Velocity offsets relative to rest wavelength. Fixed by default.
    wave : float
        Central wavelength of the profiles.
    sigma_res : float
        Instrumental resolution from the source GaussianModel.
    pure_name : str
        Base name without submodel identifiers.
    master_name : str
        Name of the source Gaussian model.

    Notes
    -----
    Cannot be combined with pipe operator ("|") in type annotations! Use 
    typing.Union or typing.Optional instead.
    """
    n_profiles: ClassVar[int] = 2

    strength_scale = Parameter(default=1, bounds=(0, None))
    
    strength_1 = Parameter(default=1,    fixed=True)
    sigma_v_1  = Parameter(default=1e-3, fixed=True)
    v_off_1    = Parameter(default=0,    fixed=True)

    strength_2 = Parameter(default=1,    fixed=True)
    sigma_v_2  = Parameter(default=1e-3, fixed=True)
    v_off_2    = Parameter(default=0,    fixed=True)

    def __init__(
        self,
        wave: float,
        name: str,
        *gs: GaussianModel,
        **kwargs,
    ):
        assert len(gs) == self.n_profiles

        super().__init__(name=name, **kwargs)

        self.pure_name:   str = name
        self.wave:      float = wave
        self.sigma_res: float = gs[0].sigma_res
        self.master_name: str = gs[0].pure_name

        self._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        self._freeze_velocity_profile(inplace=True)
        self._forget_ties(inplace=True)

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: Iterable[GaussianModel],
        freeze: bool = False,
        **kwargs,
    ) -> Self:
        model = tuple(model)
        assert len(model) == 2

        vprof = VProfileCopy2G(wave, name, *model, **kwargs)
        if freeze:
            vprof._freeze_velocity_profile(inplace=True)
            vprof._forget_ties(inplace=True)

        return vprof

class VProfileCopy3G(_VProfileCopy):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength_scale : Parameter
        Multiplicative scaling factor for all profile strengths. Bounds (0, None).
    strength_1, strength_2, strength_3 : Parameter
        Strengths of the Gaussian profiles. Fixed by default.
    sigma_v_1, sigma_v_2, sigma_v_3 : Parameter
        Velocity dispersions in $c$ units. Fixed by default.
    v_off_1, v_off_2, v_off_3 : Parameter
        Velocity offsets relative to rest wavelength. Fixed by default.
    wave : float
        Central wavelength of the profiles.
    sigma_res : float
        Instrumental resolution from the source GaussianModel.
    pure_name : str
        Base name without submodel identifiers.
    master_name : str
        Name of the source Gaussian model.

    Notes
    -----
    Cannot be combined with pipe operator ("|") in type annotations! Use 
    typing.Union or typing.Optional instead.
    """
    n_profiles: ClassVar[int] = 3

    strength_scale = Parameter(default=1, bounds=(0, None))
    
    strength_1 = Parameter(default=1,    fixed=True)
    sigma_v_1  = Parameter(default=1e-3, fixed=True)
    v_off_1    = Parameter(default=0,    fixed=True)

    strength_2 = Parameter(default=1,    fixed=True)
    sigma_v_2  = Parameter(default=1e-3, fixed=True)
    v_off_2    = Parameter(default=0,    fixed=True)

    strength_3 = Parameter(default=1,    fixed=True)
    sigma_v_3  = Parameter(default=1e-3, fixed=True)
    v_off_3    = Parameter(default=0,    fixed=True)

    def __init__(
        self,
        wave: float,
        name: str,
        *gs: GaussianModel,
        **kwargs,
    ):
        assert len(gs) == self.n_profiles

        super().__init__(name=name, **kwargs)

        self.pure_name:   str = name
        self.wave:      float = wave
        self.sigma_res: float = gs[0].sigma_res
        self.master_name: str = gs[0].pure_name

        self._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        self._freeze_velocity_profile(inplace=True)
        self._forget_ties(inplace=True)

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: Iterable[GaussianModel],
        freeze: bool = False,
        **kwargs,
    ) -> Self:
        model = tuple(model)
        assert len(model) == 3

        vprof = VProfileCopy3G(wave, name, *model, **kwargs)
        if freeze:
            vprof._freeze_velocity_profile(inplace=True)
            vprof._forget_ties(inplace=True)

        return vprof
    
class VProfileCopy4G(_VProfileCopy):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength_scale : Parameter
        Multiplicative scaling factor for all profile strengths. Bounds (0, None).
    strength_1, strength_2, strength_3, strength_4 : Parameter
        Strengths of the Gaussian profiles. Fixed by default.
    sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4 : Parameter
        Velocity dispersions in $c$ units. Fixed by default.
    v_off_1, v_off_2, v_off_3, v_off_4 : Parameter
        Velocity offsets relative to rest wavelength. Fixed by default.
    wave : float
        Central wavelength of the profiles.
    sigma_res : float
        Instrumental resolution from the source GaussianModel.
    pure_name : str
        Base name without submodel identifiers.
    master_name : str
        Name of the source Gaussian model.

    Notes
    -----
    Cannot be combined with pipe operator ("|") in type annotations! Use 
    typing.Union or typing.Optional instead.
    """
    n_profiles: ClassVar[int] = 4

    strength_scale = Parameter(default=1, bounds=(0, None))
    
    strength_1 = Parameter(default=1,    fixed=True)
    sigma_v_1  = Parameter(default=1e-3, fixed=True)
    v_off_1    = Parameter(default=0,    fixed=True)

    strength_2 = Parameter(default=1,    fixed=True)
    sigma_v_2  = Parameter(default=1e-3, fixed=True)
    v_off_2    = Parameter(default=0,    fixed=True)

    strength_3 = Parameter(default=1,    fixed=True)
    sigma_v_3  = Parameter(default=1e-3, fixed=True)
    v_off_3    = Parameter(default=0,    fixed=True)

    strength_4 = Parameter(default=1,    fixed=True)
    sigma_v_4  = Parameter(default=1e-3, fixed=True)
    v_off_4    = Parameter(default=0,    fixed=True)

    def __init__(
        self,
        wave: float,
        name: str,
        *gs: GaussianModel,
        **kwargs,
    ):
        assert len(gs) == self.n_profiles

        super().__init__(name=name, **kwargs)

        self.pure_name:   str = name
        self.wave:      float = wave
        self.sigma_res: float = gs[0].sigma_res
        self.master_name: str = gs[0].pure_name

        self._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        self._freeze_velocity_profile(inplace=True)
        self._forget_ties(inplace=True)

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: Iterable[GaussianModel],
        freeze: bool = False,
        **kwargs,
    ) -> Self:
        model = tuple(model)
        assert len(model) == 4

        vprof = VProfileCopy4G(wave, name, *model, **kwargs)
        if freeze:
            vprof._freeze_velocity_profile(inplace=True)
            vprof._forget_ties(inplace=True)

        return vprof
    
class VProfileCopy5G(_VProfileCopy):
    """
    Lorem ipsum.

    Attributes
    ----------
    strength_scale : Parameter
        Multiplicative scaling factor for all profile strengths. Bounds (0, None).
    strength_1 to strength_5 : Parameter
        Strengths of the Gaussian profiles. Fixed by default.
    sigma_v_1 to sigma_v_5 : Parameter
        Velocity dispersions in $c$ units. Fixed by default.
    v_off_1 to v_off_5 : Parameter
        Velocity offsets relative to rest wavelength. Fixed by default.
    wave : float
        Central wavelength of the profiles.
    sigma_res : float
        Instrumental resolution from the source GaussianModel.
    pure_name : str
        Base name without submodel identifiers.
    master_name : str
        Name of the source Gaussian model.

    Notes
    -----
    Cannot be combined with pipe operator ("|") in type annotations! Use 
    typing.Union or typing.Optional instead.
    """
    n_profiles: ClassVar[int] = 5

    strength_scale = Parameter(default=1, bounds=(0, None))
    
    strength_1 = Parameter(default=1,    fixed=True)
    sigma_v_1  = Parameter(default=1e-3, fixed=True)
    v_off_1    = Parameter(default=0,    fixed=True)

    strength_2 = Parameter(default=1,    fixed=True)
    sigma_v_2  = Parameter(default=1e-3, fixed=True)
    v_off_2    = Parameter(default=0,    fixed=True)

    strength_3 = Parameter(default=1,    fixed=True)
    sigma_v_3  = Parameter(default=1e-3, fixed=True)
    v_off_3    = Parameter(default=0,    fixed=True)

    strength_4 = Parameter(default=1,    fixed=True)
    sigma_v_4  = Parameter(default=1e-3, fixed=True)
    v_off_4    = Parameter(default=0,    fixed=True)

    strength_5 = Parameter(default=1,    fixed=True)
    sigma_v_5  = Parameter(default=1e-3, fixed=True)
    v_off_5    = Parameter(default=0,    fixed=True)

    def __init__(
        self,
        wave: float,
        name: str,
        *gs: GaussianModel,
        **kwargs,
    ):
        assert len(gs) == self.n_profiles

        super().__init__(name=name, **kwargs)

        self.pure_name:   str = name
        self.wave:      float = wave
        self.sigma_res: float = gs[0].sigma_res
        self.master_name: str = gs[0].pure_name

        self._adapt_to_models(*gs, tie_vel_profile=True, inplace=True)
        self._freeze_velocity_profile(inplace=True)
        self._forget_ties(inplace=True)

    @classmethod
    def from_model(
        cls,
        wave: float,
        name: str,
        model: Iterable[GaussianModel],
        freeze: bool = False,
        **kwargs,
    ) -> Self:
        model = tuple(model)
        assert len(model) == 5

        vprof = VProfileCopy5G(wave, name, *model, **kwargs)
        if freeze:
            vprof._freeze_velocity_profile(inplace=True)
            vprof._forget_ties(inplace=True)

        return vprof