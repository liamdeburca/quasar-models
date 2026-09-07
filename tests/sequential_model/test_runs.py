"""Test SequentialModel creation and evaluation with individual model types."""

import numpy as np
import pytest
from quasar_utils.setup import Info

from quasar_models import (
    BalmerModel,
    GaussianModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.balmer.continuum import BalmerContinuumTemplate
from quasar_models.balmer.series import BalmerSeriesTemplate
from quasar_models.host import HostGalaxyTemplate
from quasar_models.iron import IronTemplate
from quasar_models.modeling import SequentialModel

# Configuration
V_RES = 2.3e-4
N_WAVELENGTHS = 1000
INFO = Info()


@pytest.fixture
def wavelength_grid():
    """Create a logarithmically binned wavelength grid for testing."""
    start_wavelength = 1000.0
    return start_wavelength * (1 + V_RES) ** np.arange(N_WAVELENGTHS)


def test_from_powerlaw_model(wavelength_grid):
    """Test SequentialModel creation from a single PowerLawModel.
    
    Verifies:
    - n_submodels == 1
    - _n == 2 (flux, alpha parameters)
    - _n_free == 2
    - _n_tied == 0
    - _n_fixed == 0
    """
    x = wavelength_grid
    
    # Create PowerLawModel
    model = PowerLawModel.create(
        x0=1450.0,
        y0=1.0,
        flux=1.0,
        alpha=-1.0,
        name="powerlaw",
    )
    
    # Create SequentialModel from PowerLawModel
    seq_model = SequentialModel(model)
    
    # Verify properties
    assert seq_model.n_submodels == 1
    assert seq_model._n == 2
    assert seq_model._n_free == 2
    assert seq_model._n_tied == 0
    assert seq_model._n_fixed == 0
    
    # Test evaluate method
    params = model.parameters
    baseline = model.evaluate(x, *params)
    result = seq_model.evaluate(x, params[seq_model._free_indices])

    assert np.array_equal(baseline, result)


def test_from_gaussian_model(wavelength_grid):
    """Test SequentialModel creation from a single GaussianModel.
    
    Verifies:
    - n_submodels == 1
    - _n == 3 (strength, fwhm_v, v_off parameters)
    - _n_free == 3
    - _n_tied == 0
    - _n_fixed == 0
    """
    x = wavelength_grid
    
    # Create GaussianModel
    model = GaussianModel.create(
        wave=1549.0,
        sigma_res=V_RES,
        strength=1.0,
        fwhm_v=1e-4,
        v_off=0.0,
        name="gaussian",
    )
    
    # Create SequentialModel from GaussianModel
    seq_model = SequentialModel(model)
    
    # Verify properties
    assert seq_model.n_submodels == 1
    assert seq_model._n == 3
    assert seq_model._n_free == 3
    assert seq_model._n_tied == 0
    assert seq_model._n_fixed == 0

    # Test evaluate method
    params = model.parameters
    baseline = model.evaluate(x, *params)
    result = seq_model.evaluate(x, params[seq_model._free_indices])

    assert np.array_equal(baseline, result)


def test_from_iron_model(wavelength_grid):
    """Test SequentialModel creation from a single IronModel.
    
    Verifies:
    - n_submodels == 1
    - _n == 5 (flux, fwhm, split, left, right parameters)
    - _n_free == 2 (flux, fwhm parameters)
    - _n_tied == 0
    - _n_fixed == 3 (split, left, right parameters)
    """
    x = wavelength_grid
    
    # Load and adapt template
    _template = IronTemplate.load(path="vw2001", info=INFO)
    template = _template.createLogspace(sigma_res=V_RES, xr=x, keep_x=True)
    
    scale = INFO.iron["scale"]
    
    # Create IronModel
    model = IronModel.create(
        flux=1.0,
        fwhm=template.fwhm[10],  # Use a mid-range FWHM
        scale=scale,
        template=template,
        name="iron",
        allow_interp_fitting=False,
    )
    
    # Create SequentialModel from IronModel
    seq_model = SequentialModel(model)
    
    # Verify properties
    assert seq_model.n_submodels == 1
    assert seq_model._n == 5
    assert seq_model._n_free == 2
    assert seq_model._n_tied == 0
    assert seq_model._n_fixed == 3
    
    # Test evaluate method
    params = model.parameters
    baseline = model.evaluate(x, *params)
    result = seq_model.evaluate(x, params[seq_model._free_indices])

    assert np.array_equal(baseline, result)
    

def test_from_balmer_model(wavelength_grid):
    """Test SequentialModel creation from a single BalmerModel.
    
    Verifies:
    - n_submodels == 1
    - _n == 3 (flux, fwhm, ratio parameters)
    - _n_free == 2 (flux, fwhm parameters)
    - _n_tied == 0
    - _n_fixed == 1 (ratio parameter)
    """
    x = wavelength_grid
    
    # Load templates
    continuum_template = BalmerContinuumTemplate.load_from_cache(
        temp=INFO.balmer.temp,
        tau=INFO.balmer.tau,
        scale=INFO.balmer.scale,
        info=INFO,
    )
    series_template = BalmerSeriesTemplate.load_from_cache(
        name="sh1995",
        temp=INFO.balmer.temp,
        dens=INFO.balmer.dens,
        n_u_range=(INFO.balmer.n_u_min, INFO.balmer.n_u_max),
        info=INFO,
    )
    
    # Create BalmerModel
    model = BalmerModel.create(
        flux=INFO.balmer.flux,
        fwhm=INFO.balmer.fwhm,  # Use a mid-range FWHM
        ratio=INFO.balmer.ratio,
        edge=INFO.balmer.edge,
        continuum_template=continuum_template,
        series_template=series_template,
        info=INFO,
        name="sh1995",
        allow_interp_fitting=False,
    )
    
    # Create SequentialModel from BalmerModel
    seq_model = SequentialModel(model)
    
    # Verify properties
    assert seq_model.n_submodels == 1
    assert seq_model._n == 3
    assert seq_model._n_free == 3
    assert seq_model._n_tied == 0
    assert seq_model._n_fixed == 0
    
    # Test evaluate method
    params = model.parameters
    baseline = model.evaluate(x, *params)
    result = seq_model.evaluate(x, params[seq_model._free_indices])

    assert np.array_equal(baseline, result)


def test_from_host_galaxy_model(wavelength_grid):
    """Test SequentialModel creation from a single HostGalaxyModel.
    
    Verifies:
    - n_submodels == 1
    - _n == 2 (flux, fwhm parameters)
    - _n_free == 1 (flux parameter)
    - _n_tied == 0
    - _n_fixed == 1 (fwhm parameter)
    """
    x = wavelength_grid
    
    # Load template
    _template = HostGalaxyTemplate.load_from_cache(
        name=INFO.host.sources[0],
        age=INFO.host.ages[0],
        info=INFO,
    )
    template = _template.createLogspace(sigma_res=V_RES, xr=x, keep_x=True)
    
    # Create HostGalaxyModel
    model = HostGalaxyModel.create(
        flux=INFO.host.flux,
        fwhm=INFO.host.fwhm,  # Use a mid-range FWHM
        template=template,
        info=INFO,
        name=INFO.host.sources[0],
        age=INFO.host.ages[0],
        allow_interp_fitting=False,
    )
    
    # Create SequentialModel from HostGalaxyModel
    seq_model = SequentialModel(model)
    
    # Verify properties
    assert seq_model.n_submodels == 1
    assert seq_model._n == 2
    assert seq_model._n_free == 1
    assert seq_model._n_tied == 0
    assert seq_model._n_fixed == 1

    # Test evaluate method
    params = model.parameters
    baseline = model.evaluate(x, *params)
    result = seq_model.evaluate(x, params[seq_model._free_indices])

    assert np.array_equal(baseline, result)