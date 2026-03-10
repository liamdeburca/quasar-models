from dataclasses import dataclass
from typing import Union, Optional, Iterable, Callable
from numpy import ndarray
from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from ...utils import powerlaw_evaluate, mean_func, sigma_func
from ..utils.template import Template

from ...typing_.arrays import FloatVector
from ...typing_.misc.astropy import ModelLike
from ...typing_.misc.custom_models import PowerLawLike, PowerLawModelLike, \
    GaussianModelLike, TemplateModelLike

### Individual models

@dataclass
class PowerLawDC:
    """
    DataClass instance which contains the fit parameters for a single power law
    model instance. 
    """
    x0: float
    flux: float
    alpha: float

    def __call__(
        self, 
        x: Union[float, FloatVector],
    ) -> Union[float, FloatVector]:
        return powerlaw_evaluate(x, self.x0, self.flux, self.alpha)
            
    @staticmethod
    def fromPowerLaw(model: PowerLawLike) -> 'PowerLawDC':
        """
        Converts an 'AstroPy' continuum model, either 'PowerLawModel' or 
        'PLWiggle' instance, into a 'PowerLawDC' instance. In the latter case,
        the wiggle model is converted into the corresponding power law model.
        """
        if (model.__class__.name == 'PowerLawModel') \
            or (model.__class__.name == 'PLWiggle'):
            return PowerLawDC(
                model.x0,
                model.flux.value,
                model.alpha.value,
            )
        else:
            msg = "Model must be an instance of either  'PowerLawModel' or a " \
            "'PLWiggle' class!"
            raise ValueError(msg)
    
    @property
    def inv(
        self,
    ) -> Callable[[Union[float, FloatVector]], Union[float, FloatVector]]:
        from functools import partial
        return partial(
            powerlaw_evaluate,
            x0 = self.x0,
            flux = 1 / self.flux,
            alpha = -self.alpha,
        )
    
@dataclass 
class PLWiggleDC:
    """
    DataClass instance which contains the fit parameters for a single power law 
    wiggle model instance. 
    """
    x0: float
    flux: float
    alpha: float
    flux_old: float
    alpha_old: float

    def __call__(
        self, 
        x: Union[float, FloatVector],
    ) -> Union[float, FloatVector]:
        return self.new(x) - self.old(x)
    
    @staticmethod
    def fromPowerLaw(model: PowerLawLike) -> 'PLWiggleDC':
        """
        Converts an 'AstroPy' continuum model, either 'PowerLawModel' or 
        'PLWiggle' instance, into a 'PowerLawDC' instance. In the former case,
        the power law model is converted into the correpsponding wiggle model.
        """
        if (model.__class__.name == 'PLWiggle'):
            return PLWiggleDC(
                model.x0,
                model.flux.value,
                model.alpha.value,
                model.flux_old,
                model.alpha_old,
            )
        elif (model.__class__.name == 'PowerLawModel'):
            return PLWiggleDC(
                model.x0,
                model.flux.value,
                model.alpha.value,
                model.flux.value,
                model.alpha.value,
            )
        else:
            msg = "Model must be an instance of either  'PowerLawModel' or a " \
            "'PLWiggle' class!"
            raise ValueError(msg)

    @property
    def old(self) -> PowerLawDC:
        return PowerLawDC(self.x0, self.flux_old, self.alpha_old)
    
    @property
    def new(self) -> PowerLawDC:
        return PowerLawDC(self.x0, self.flux_new, self.alpha_new)
    
@dataclass
class TemplateDC:
    """
    DataClass instance which contains the fit parameters for a Template model
    instance. 
    """
    flux:  float
    fwhm:  float
    split: float
    left:  float
    right: float

    scale: float
    template: Template

    def __call__(
        self,
        x: Union[float, FloatVector]
    ) -> Union[float, FloatVector]:
        from ...utils import template_evaluate
        return template_evaluate(
            x, 
            self.flux, self.fwhm, self.split, self.left, self.right,
            self.scale, self.template,
        )
    
    def fromTemplate(model: TemplateModelLike) -> 'TemplateDC':
        """
        Converts an 'AstroPy' 'TemplateModel' instance into the corresponding
        dataclass instance. 
        """
        return TemplateDC(
            model.flux.value,
            model.fwhm.value,
            model.split.value,
            model.left.value,
            model.right.value,
            model.scale,
            model.template,
        )
    
@dataclass
class MultipleTemplatesDC:
    flux:  ndarray[float]
    fwhm:  ndarray[float]
    split: ndarray[float]
    left:  ndarray[float]
    right: ndarray[float]

    scale:    ndarray[float]
    template: list[Template]

    def __call__(
        self,
        x: Union[float, FloatVector],
    ) -> Union[float, FloatVector]:
        from ...utils import template_evaluate
        return sum([
            template_evaluate(x, *args, validate=False) \
            for args in zip(
                self.flux, self.fwhm, self.split, self.left, self.right, 
                self.scale, self.template,
            )
        ])

    @staticmethod
    def fromTemplates(
        models: Union[TemplateModelLike, ModelLike],
        simplify: bool = False,
    ) -> Union['TemplateDC', 'MultipleTemplatesDC']:
        """
        Converts 'AstroPy' 'TemplateModel' instances into the corresponding
        dataclass instance. 
        """
        from numpy import array

        if not isinstance(models, list):
            models = [models] if models.n_submodels == 1 else list(models)

        if (len(models) == 1) and simplify:
            return TemplateDC.fromTemplate(models[0])
        
        flux:  list[float] = []
        fwhm:  list[float] = []
        split: list[float] = []
        left:  list[float] = []
        right: list[float] = []

        scale:    list[float] = []
        template: list[Template] = []

        for model in models:
            flux .append(model.flux.value)
            fwhm .append(model.fwhm.value)
            split.append(model.split.value)
            left .append(model.left.value)
            right.append(model.right.value)

            scale.append(model.scale)
            template.append(model.template)

        return MultipleTemplatesDC(
            array(flux),
            array(fwhm),
            array(split),
            array(left),
            array(right),
            array(scale),
            template,
        )
@dataclass
class GaussianDC:
    """
    DataClass instance which contains the fit parameters for a Gaussian model
    instance. 
    """
    wave:      float
    sigma_res: float
    strength:  float
    sigma_v:   float
    v_off:     float

    def __call__(
        self, 
        x: Union[float, FloatVector],
    ) -> Union[float, FloatVector]:
        from ...utils import gaussian_evaluate
        return gaussian_evaluate(
            x, self.wave, self.strength, self.sigma_v, self.v_off, 
            self.sigma_res,
        )

    @staticmethod
    def fromGaussian(model: GaussianModelLike) -> 'GaussianDC':
        """
        Converts an 'AstroPy' 'GaussianModel' instance into the a corresponding
        dataclass instance. 
        """
        return GaussianDC(
            model.wave,
            model.sigma_res,
            model.strength.value,
            model.sigma_v.value,
            model.v_off.value,
        )
    
    @property
    def n_submodels(self) -> int:
        return 1
    
    @property 
    def mu(self) -> float:
        return self.wave * (1 + self.v_off)
    
    @property
    def sigma(self) -> float:
        return sigma_func(self.wave, self.sigma_v, self.v_off, self.sigma_res)
    
    @property
    def peak(self) -> float:
        from numpy import pi
        return self.strength / (self.sigma * (2 * pi)**0.5)
    
    @property
    def dist(self) -> rv_continuous_frozen:
        if not hasattr(self, '_dist'):
            self._dist = norm(loc=self.mu, scale=self.sigma)
        return self._dist
    
    @property
    def pdf(
        self,
    ) -> Callable[[Union[float, FloatVector]], Union[float, FloatVector]]:
        return self.dist.pdf
    
    @property
    def cdf(self) -> Callable[[Union[float, FloatVector]], Union[float, FloatVector]]:
        return self.dist.cdf
    
    @property
    def isf(self) -> Callable[[Union[float, FloatVector]], Union[float, FloatVector]]:
        return self.dist.isf

        
### Combinations of gaussians

@dataclass
class MultipleGaussiansDC:
    """
    DataClass instance which contains the fit parameters for multiple Gaussian 
    model instances, i.e. a compound model. 
    """
    wave: ndarray[float]
    sigma_res: float

    strength: ndarray[float]
    sigma_v: ndarray[float]
    v_off: ndarray[float]

    def __call__(
        self, 
        x: Union[float, FloatVector],
    ) -> Union[float, FloatVector]:
        from ...utils import gaussian_evaluate
        return sum([
            gaussian_evaluate(x, *args, self.sigma_res) \
            for args in zip(
                self.wave, self.strength, self.sigma_v, self.v_off,
            )
        ])    
    @staticmethod
    def fromGaussians(
        models: Union[list[Union[GaussianDC, GaussianModelLike]], ModelLike],
        simplify: bool = False,
    ) -> Union['GaussianDC', 'MultipleGaussiansDC']:
        """
        Converts an 'AstroPy' model into the corresponding dataclass instance. 

        Notes
        -----
        By default, a model containing a single 'GaussianModel' instance is 
        still converted into a 'MultipleGaussianDC' instance. If 'simplify' is 
        set to True, a 'GaussianDC' class is returned instead. 
        """
        from numpy import array

        if not isinstance(models, list):
            models = [models] if (models.n_submodels == 1) else list(models)

        if (len(models) == 1) and simplify:
            return GaussianDC.fromGaussian(models[0])
        
        return MultipleGaussiansDC(
            array([m.wave           for m in models]),
            models[0].sigma_res,
            array([m.strength.value for m in models]),
            array([m.sigma_v.value  for m in models]),
            array([m.v_off.value    for m in models]),
        )

    def order(self) -> None:
        from numpy import argsort

        indices = argsort(self.wave * (1 + self.v_off))
        self.wave = self.wave[indices]
        self.strength = self.strength[indices]
        self.sigma_v = self.sigma_v[indices]
        self.v_off = self.v_off[indices]

    def splitAtWave(
        self, 
        wave: Union[float, Iterable[float]],
        simplify: bool = True
    ) -> tuple[Optional[GaussianDC], Optional[GaussianDC]]:
        from numpy import isin, argwhere

        _in = argwhere(isin(self.wave, wave)).flatten()
        _out = argwhere(isin(self.wave, wave, invert=True)).flatten()
        
        # Targeted models
        if _in.size == 0:
            gaussians = None
        elif _in.size == 1:
            idx = _in[0]
            if simplify:
                gaussians = GaussianDC(
                    self.wave[idx], 
                    self.sigma_res,
                    self.strength[idx],
                    self.sigma_v[idx],
                    self.v_off[idx],
                )
            else:
                sel = slice(idx, idx+1)
                gaussians = MultipleGaussiansDC(
                    self.wave[sel], 
                    self.sigma_res,
                    self.strength[sel],
                    self.sigma_v[sel],
                    self.v_off[sel],
                )
        else:
            gaussians = MultipleGaussiansDC(
                self.wave[_in],
                self.sigma_res,
                self.strength[_in],
                self.sigma_v[_in],
                self.v_off[_in],
            )

        # Other models
        if _out.size == 0:
            other_gaussians = None
        elif _out.size == 1:
            idx = _out[0]
            if simplify:
                other_gaussians = GaussianDC(
                    self.wave[idx], 
                    self.sigma_res,
                    self.strength[idx],
                    self.sigma_v[idx],
                    self.v_off[idx],
                )
            else:
                sel = slice(idx, idx+1)
                other_gaussians = MultipleGaussiansDC(
                    self.wave[sel], 
                    self.sigma_res,
                    self.strength[sel],
                    self.sigma_v[sel],
                    self.v_off[sel],
                )
        else:
            other_gaussians = MultipleGaussiansDC(
                self.wave[_out],
                self.sigma_res,
                self.strength[_out],
                self.sigma_v[_out],
                self.v_off[_out],
            )

        return gaussians, other_gaussians
        
    @property
    def n_submodels(self) -> int:
        return len(self.wave)
    
    @property
    def mu(self) -> FloatVector:
        return mean_func(self.wave, self.v_off)
    
    @property
    def sigma(self) -> FloatVector:
        return sigma_func(self.wave, self.sigma_v, self.v_off, self.sigma_res)
    
    @property
    def weight(self) -> FloatVector:
        return self.strength / self.strength.sum()
    
    @property
    def peak(self) -> Optional[float]:
        from math import pi
        return self.strength[0] / (self.sigma[0] * (2 * pi)**0.5) \
            if self.n_submodels == 1 \
            else None
            
    @property
    def dist(self) -> list[rv_continuous_frozen]:
        if not hasattr(self, '_dist'):        
            self._dist = [norm(*args) for args in zip(self.mu, self.sigma)]
        return self._dist
    
    @property
    def pdf(
        self,
    ) -> Callable[[Union[float, FloatVector]], Union[float, FloatVector]]:

        if not hasattr(self, '_pdf'):
            def pdf(x: Union[float, FloatVector]) -> Union[float, FloatVector]:
                from numpy import stack
                return stack([
                    w * d.pdf(x) for w, d in zip(self.weight, self.dist)
                ]).sum(axis=0)
            
            self._pdf = pdf

        return self._pdf
    
    @property
    def cdf(
        self,
    ) -> Callable[[Union[float, FloatVector]], Union[float, FloatVector]]:

        if not hasattr(self, '_cdf'):
        
            def cdf(x: Union[float, FloatVector]) -> Union[float, FloatVector]:
                from numpy import stack
                return stack([
                    w * d.cdf(x) for w, d in zip(self.weight, self.dist)
                ]).sum(axis=0)

            self._cdf = cdf

        return self._cdf
    
    @property
    def isf(
        self,
    ) -> Optional[Callable[[Union[float, FloatVector]], Union[float, FloatVector]]]:
        
        isf = self.dist[0].isf \
            if self.n_submodels == 1 \
            else None
        
        return isf

### Bootstrap sample: Powerlaw and MultipleGaussians

@dataclass
class Sample:
    """
    Contains a model sample, i.e. a combination of emission line models and 
    optionally a power law model. 
    """
    pl: Optional[PowerLawDC] = None
    fe: Optional[MultipleTemplatesDC] = None
    em: Optional[MultipleGaussiansDC] = None
    weight: float = 1.0

    @staticmethod
    def fromCompoundModel(model: ModelLike) -> 'Sample':
        """
        Converts an 'AstroPy' compound model into a 'Sample' instance.

        Notes
        -----
        Even if the input model only contains a single emission line model, i.e.
        a Gaussian function, this is still converted into a 'MultipleGaussianDC'
        instance.
        """
        pl_models: list = []
        fe_models: list = []
        em_models: list = []

        submodels = [model] if model.n_submodels == 1 else list(model)

        for submodel in submodels:
            if submodel.__class__.name in ('PowerLawModel', 'PLWiggle'):
                pl_models.append(submodel)
                continue
            
            if submodel.__class__.name == 'TemplateModel':
                fe_models.append(submodel)
                continue

            if submodel.__class__.name == 'GaussianModel':
                em_models.append(submodel)
                continue

            if submodel.__class__.name.startswith('VProfileCopy'):
                em_models += submodel.splitInfoGaussians()
                continue

        pl = None
        if len(pl_models) == 1: pl = PowerLawDC.fromPowerLaw(pl_models[0])
        
        fe = None
        if len(fe_models) != 0: fe = MultipleTemplatesDC.fromTemplates(
            fe_models, simplify=False,
        )
            
        em = None
        if len(em_models) != 0: em = MultipleGaussiansDC.fromGaussians(
            em_models, simplify=False,
        )
            
        return Sample(pl=pl, fe=fe, em=em)
    
    @staticmethod
    def fromParamVector(
        model: ModelLike,
        p: FloatVector,
        weight: float = 1.0,
    ) -> 'Sample':
        """
        Converts a 'NestedResult' instance into a 'Sample' instance using an 
        'AstroPy' model as a template. 

        Notes
        -----
        Even if the input model only contains a single emission line model, i.e.
        a Gaussian function, this is still converted into a 'MultipleGaussianDC'
        instance.
        """
        from numpy import array

        pl = None

        fe_flux:     list[float] = []
        fe_fwhm:     list[float] = []
        fe_split:    list[float] = []
        fe_left:     list[float] = []
        fe_right:    list[float] = []
        fe_scale:    list[float] = []
        fe_template: list = []

        em_wave:      list[float] = []
        em_sigma_res: list[float] = []
        em_strength:  list[float] = []
        em_sigma_v:   list[float] = []
        em_v_off:     list[float] = []

        submodels = [model] if model.n_submodels == 1 else list(model)

        count: int = 0
        for submodel in submodels:
            match submodel.__class__.name:
                case 'PowerLawModel':                    
                    new_count = count + 2
                    pl = PowerLawDC(
                        submodel.x0, *p[count:new_count],
                    )

                case 'TemplateModel':
                    new_count = count + 5

                    fe_flux .append(p[count])
                    fe_fwhm .append(p[count+1])
                    fe_split.append(p[count+2])
                    fe_left .append(p[count+3])
                    fe_right.append(p[count+4])

                    fe_scale.append(submodel.scale)
                    fe_template.append(submodel.template)

                case 'GaussianModel':
                    new_count = count + 3

                    em_wave     .append(submodel.wave)
                    em_sigma_res.append(submodel.sigma_res)
                    em_strength .append(p[count])
                    em_sigma_v  .append(p[count+1])
                    em_v_off    .append(p[count+2])

            count = new_count
        
        fe = None
        if len(fe_flux) != 0: fe = MultipleTemplatesDC(
            array(fe_flux),
            array(fe_fwhm),
            array(fe_split),
            array(fe_left),
            array(fe_right),
            array(fe_scale),
            fe_template,
        )
            
        em = None
        if len(em_wave) != 0: em = MultipleGaussiansDC(
            array(em_wave),
            em_sigma_res[0],
            array(em_strength),
            array(em_sigma_v),
            array(em_v_off),
        )
            
        return Sample(pl=pl, fe=fe, em=em, weight=weight)

    def order(self) -> 'Sample':
        _ = self.em.order()
        return self

    def splitAtWave(
        self, 
        wave: Union[float, Iterable[float]],
        simplify: bool = True,
    ) -> tuple['Sample', 'Sample']:
        em, other_em = self.em.splitAtWave(wave, simplify=simplify)
        return (
            Sample(pl=self.pl, fe=self.fe, em=em),
            Sample(pl=self.pl, fe=self.fe, em=other_em),
        )