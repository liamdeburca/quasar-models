"""
Test script for copy() method of custom AstropyModel classes.

Tests verify that the copy() method correctly preserves:
- Model names (identical in value)
- Meta dictionaries (identical in value)
- Parameter values, bounds, fixed status, and tied values
"""
import numpy as np
from astropy.modeling import Fittable1DModel, CompoundModel
from quasar_models.tying import IdenticalTie

from quasar_models import PowerLawModel, GaussianModel, BalmerModel, IronModel, HostGalaxyModel
from quasar_utils.setup import Info

N_INSTANCES = 100
N_SUBMODELS = 10
INFO = Info()

def generate_random_bounds(
    lower: float | None, 
    upper: float | None, 
) -> tuple[float | None, float | None]:
    """
    Generate random bounds within the specified range.
    
    Respects None bounds (unbounded limits) and returns valid bound pairs.
    """
    if lower is None and upper is None:
        return (None, None)
    
    if lower is None:
        return (None, upper)
    
    if upper is None:
        return (lower, None)
    
    if lower == upper:
        return (lower, upper)
    
    return tuple(sorted((np.random.uniform(lower, upper, size=2))))

def sample_value_from_bounds(
    bounds: tuple[float | None, float | None]
) -> float:
    """
    Sample a value uniformly from the given bounds.
    
    Handles unbounded cases by defaulting to reasonable ranges.
    """
    lower, upper = bounds
    
    if lower is None and upper is None:
        return np.random.uniform(-1, 1)
    
    if lower is None:
        return np.random.uniform(upper - 2, upper)
    
    if upper is None:
        return np.random.uniform(lower, lower + 2)
    
    return np.random.uniform(lower, upper)


def verify_model_copy(original: Fittable1DModel) -> None:
    """
    Verify that copy() correctly preserves all model attributes.
    
    Asserts that the copied model has identical names, meta dictionaries,
    and parameter values, bounds, fixed status, and tied values.
    """
    copied = original.copy()
    assert original is not copied
    assert original == copied
        
    for param_name in original.param_names:
        original_param = getattr(original, param_name)
        copied_param = getattr(copied, param_name)
        
        assert original_param.value == copied_param.value, \
            f"Parameter '{param_name}' values differ: " \
            f"{original_param.value} != {copied_param.value}"
        
        assert original_param.bounds == copied_param.bounds, \
            f"Parameter '{param_name}' bounds differ: " \
            f"{original_param.bounds} != {copied_param.bounds}"
        
        assert original_param.fixed == copied_param.fixed, \
            f"Parameter '{param_name}' fixed status differs: " \
            f"{original_param.fixed} != {copied_param.fixed}"
        
        assert original_param.tied == copied_param.tied, \
            f"Parameter '{param_name}' tied status differs: " \
            f"{original_param.tied} != {copied_param.tied}"

def apply_random_ties(model: Fittable1DModel) -> None:
    """
    Randomly assign IdenticalTie to parameters with 50% probability each.
    
    For tied parameters, target_parameter is set to the parameter name.
    """
    for param_name in model.param_names:
        if bool(np.random.choice([True, False])):
            param = getattr(model, param_name)
            param.tied = IdenticalTie(target_name='some_other_model', target_parameter=param_name)

def _generate_random_suffix() -> str:
    from string import ascii_letters
    return ''.join(np.random.choice(list(ascii_letters), size=8))

def _generate_random_powerlaw_model() -> PowerLawModel:
    flux_bounds = generate_random_bounds(0.0, 100.0)
    alpha_bounds = generate_random_bounds(-10.0, 10.0)
    
    flux_value = sample_value_from_bounds(flux_bounds)
    alpha_value = sample_value_from_bounds(alpha_bounds)
    
    x0 = np.random.uniform(1000, 4000)
    y0 = np.random.uniform(0, 1)
    
    model = PowerLawModel.create(
        x0=x0,
        y0=y0,
        flux=flux_value,
        alpha=alpha_value,
        name='powerlaw_' + _generate_random_suffix(),
    )
    
    model.flux.bounds = flux_bounds
    model.alpha.bounds = alpha_bounds
    
    model.flux.fixed = bool(np.random.choice([True, False]))
    model.alpha.fixed = bool(np.random.choice([True, False]))

    apply_random_ties(model)

    return model

def _generate_random_gaussian_model() -> GaussianModel:
    strength_bounds = generate_random_bounds(0.0, 100.0)
    sigma_v_bounds = generate_random_bounds(0.0, 1e-2)
    v_off_bounds = generate_random_bounds(-1e-2, 1e-2)
    
    strength_value = sample_value_from_bounds(strength_bounds)
    sigma_v_value = sample_value_from_bounds(sigma_v_bounds)
    v_off_value = sample_value_from_bounds(v_off_bounds)
    
    model = GaussianModel.create(
        wave=np.random.uniform(1000, 10000),
        sigma_res=np.random.uniform(1e-4, 1e-2),
        strength=strength_value,
        sigma_v=sigma_v_value,
        v_off=v_off_value,
        n_sigmas=3.0,
        name='gaussian_' + _generate_random_suffix(),
    )
    model.strength.bounds = strength_bounds
    model.sigma_v.bounds = sigma_v_bounds
    model.v_off.bounds = v_off_bounds
    
    model.strength.fixed = bool(np.random.choice([True, False]))
    model.sigma_v.fixed = bool(np.random.choice([True, False]))
    model.v_off.fixed = bool(np.random.choice([True, False]))
    
    apply_random_ties(model)

    return model

def _generate_random_balmer_model() -> BalmerModel:
    flux_bounds = generate_random_bounds(0.0, 100.0)
    fwhm_bounds = generate_random_bounds(0.0, 1e-2)
    ratio_bounds = generate_random_bounds(0.0, 2.0)
    
    flux_value = sample_value_from_bounds(flux_bounds)
    fwhm_value = sample_value_from_bounds(fwhm_bounds)
    ratio_value = sample_value_from_bounds(ratio_bounds)

    model = BalmerModel.create(
        flux=flux_value,
        fwhm=fwhm_value,
        edge=INFO.balmer.edge,
        info=INFO,
        ratio=ratio_value,
        name=INFO.balmer.source,
        temp=INFO.balmer.temp,
        tau=INFO.balmer.tau,
        scale=INFO.balmer.scale,
        dens=INFO.balmer.dens,
        n_u_range=(INFO.balmer.n_u_min, INFO.balmer.n_u_max),
        allow_interp_fitting=INFO.balmer.allow_interp_fitting,
    )
    model.flux.bounds = flux_bounds
    model.fwhm.bounds = fwhm_bounds
    model.ratio.bounds = ratio_bounds
    
    model.flux.fixed = bool(np.random.choice([True, False]))
    model.fwhm.fixed = bool(np.random.choice([True, False]))
    model.ratio.fixed = bool(np.random.choice([True, False]))
    
    apply_random_ties(model)

    return model

def _generate_random_iron_model() -> IronModel:
    flux_bounds = generate_random_bounds(0.0, 100.0)
    fwhm_bounds = generate_random_bounds(0.0, 1e-2)
    split_bounds = generate_random_bounds(0.0, 1e-2)
    left_bounds = generate_random_bounds(0.0, 1.0)
    right_bounds = generate_random_bounds(0.0, 1.0)
    
    flux_value = sample_value_from_bounds(flux_bounds)
    fwhm_value = sample_value_from_bounds(fwhm_bounds)
    split_value = sample_value_from_bounds(split_bounds)
    left_value = sample_value_from_bounds(left_bounds)
    right_value = sample_value_from_bounds(right_bounds)

    model = IronModel.create(
        flux_value, fwhm_value,
        scale=INFO.iron.scale,
        info=INFO,
        split=split_value,
        left=left_value,
        right=right_value,
        allow_interp_fitting=bool(np.random.choice([True, False])),
        name='vw2001',
    )
    model.flux.bounds = flux_bounds
    model.fwhm.bounds = fwhm_bounds
    model.split.bounds = split_bounds
    model.left.bounds = left_bounds
    model.right.bounds = right_bounds
    
    model.flux.fixed = bool(np.random.choice([True, False]))
    model.fwhm.fixed = bool(np.random.choice([True, False]))
    model.split.fixed = bool(np.random.choice([True, False]))
    model.left.fixed = bool(np.random.choice([True, False]))
    model.right.fixed = bool(np.random.choice([True, False]))
    
    apply_random_ties(model)

    return model

def _generate_random_host_galaxy_model() -> HostGalaxyModel:
    flux_bounds = generate_random_bounds(0.0, 100.0)
    fwhm_bounds = generate_random_bounds(0.0, 1e-2)
    
    flux_value = sample_value_from_bounds(flux_bounds)
    fwhm_value = sample_value_from_bounds(fwhm_bounds)

    model = HostGalaxyModel.create(
        flux=flux_value,
        fwhm=fwhm_value,
        info=INFO,
        allow_interp_fitting=bool(np.random.choice([True, False])),
        name='bc2003',
        age=8e9,
    )
    model.flux.bounds = flux_bounds
    model.fwhm.bounds = fwhm_bounds
    
    model.flux.fixed = bool(np.random.choice([True, False]))
    model.fwhm.fixed = bool(np.random.choice([True, False]))
    
    apply_random_ties(model)

    return model

###

def verify_compound_model_copy(original_compound) -> None:
    """
    Verify that copy() correctly preserves all submodels in a compound model.
    
    Asserts that the copied compound model has the same number of submodels
    and each submodel is copied correctly with identical attributes.
    """
    copied_compound = original_compound.copy()
    
    assert original_compound.n_submodels == copied_compound.n_submodels, \
        f"Number of submodels differs: {original_compound.n_submodels} != {copied_compound.n_submodels}"
    
    for o, c in zip(original_compound, copied_compound):
        assert o.name == c.name, \
            f"Submodel names differ: {o.name} != {c.name}"

        for key, value in o.meta.items():
            assert value == c.meta[key], \
                f"Submodel meta '{key}' differs: {value} != {c.meta[key]}"
                
        for param_name in o.param_names:
            orig_param = getattr(o, param_name)
            copy_param = getattr(c, param_name)
            
            assert orig_param.value == copy_param.value, \
                f"Submodel parameter '{param_name}' value differs"
            assert orig_param.bounds == copy_param.bounds, \
                f"Submodel parameter '{param_name}' bounds differ"
            assert orig_param.fixed == copy_param.fixed, \
                f"Submodel parameter '{param_name}' fixed differs"
            assert orig_param.tied == copy_param.tied, \
                f"Submodel parameter '{param_name}' tied differs"

def _generate_random_compound_model() -> CompoundModel:
    submodels = []
    for _ in range(N_SUBMODELS):
        match np.random.choice(['pl', 'fe', 'ba', 'hg', 'em']):
            case 'pl':
                submodel = _generate_random_powerlaw_model()
            case 'fe':
                submodel = _generate_random_iron_model()
            case 'ba':
                submodel = _generate_random_balmer_model()
            case 'hg':
                submodel = _generate_random_host_galaxy_model()
            case 'em':
                submodel = _generate_random_gaussian_model()

        submodels.append(submodel)

    return sum(submodels[1:], submodels[0])

###

def test_powerlaw_model_copy() -> None:
    """
    Verify copy() preserves names, meta, and all parameter attributes for PowerLawModel.
    
    Generates N instances with random bounds, values, and fixed status.
    For each instance, verifies that copy() produces identical names, meta dictionaries,
    and parameter values, bounds, fixed status, and tied values.
    """    
    for _ in range(N_INSTANCES):
        model = _generate_random_powerlaw_model()
        verify_model_copy(model)

def test_gaussian_model_copy() -> None:
    """
    Verify copy() preserves names, meta, and all parameter attributes for GaussianModel.
    
    Generates N instances with random bounds, values, and fixed status.
    For each instance, verifies that copy() produces identical names, meta dictionaries,
    and parameter values, bounds, fixed status, and tied values.
    """    
    for _ in range(N_INSTANCES):
        model = _generate_random_gaussian_model()
        verify_model_copy(model)

def test_balmer_model_copy() -> None:
    """
    Verify copy() preserves names, meta, and all parameter attributes for BalmerModel.
    
    Generates N instances with random bounds, values, and fixed status.
    For each instance, verifies that copy() produces identical names, meta dictionaries,
    and parameter values, bounds, fixed status, and tied values.
    """
    for _ in range(N_INSTANCES):
        model = _generate_random_balmer_model()
        verify_model_copy(model)

def test_iron_model_copy() -> None:
    """
    Verify copy() preserves names, meta, and all parameter attributes for IronModel.
    
    Generates N instances with random bounds, values, and fixed status.
    For each instance, verifies that copy() produces identical names, meta dictionaries,
    and parameter values, bounds, fixed status, and tied values.
    """
    for _ in range(N_INSTANCES):
        model = _generate_random_iron_model()
        verify_model_copy(model)

def test_host_galaxy_model_copy() -> None:
    """
    Verify copy() preserves names, meta, and all parameter attributes for HostGalaxyModel.
    
    Generates N instances with random bounds, values, and fixed status.
    For each instance, verifies that copy() produces identical names, meta dictionaries,
    and parameter values, bounds, fixed status, and tied values.
    """
    for _ in range(N_INSTANCES):
        model = _generate_random_host_galaxy_model()
        verify_model_copy(model)

def test_compound_model_copy() -> None:
    """
    Verify copy() preserves all submodels in randomly constructed compound models.
    
    Generates N_INSTANCES compound models, each with N_SUBMODELS random submodels
    of mixed types. For each compound model, verifies that copy() produces an
    identical compound model with all submodels correctly copied.
    """
    for _ in range(N_INSTANCES):
        compound_model = _generate_random_compound_model()
        verify_compound_model_copy(compound_model)