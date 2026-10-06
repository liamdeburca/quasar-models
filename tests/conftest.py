from math import floor, log
from pathlib import Path
from shutil import rmtree

import pytest
from astropy.units import Unit
from numpy import arange
from numpy.random import Generator, default_rng
from quasar_typing.numpy import FloatVector
from quasar_utils.setup import Info

from quasar_models.balmer import (
    BalmerContinuumTemplate,
    BalmerModel,
    BalmerSeriesTemplate,
)
from quasar_models.continuum import PowerLawModel
from quasar_models.host import HostGalaxyModel, HostGalaxyTemplate
from quasar_models.iron import IronModel, IronTemplate
from quasar_models.line import GaussianModel, VProfileCopy1G


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--deriv-epsilon",
        type=float,
        default=1e-6,
        help="Epsilon for numerical derivatives",
    )
    parser.addoption(
        "--deriv-its",
        type=int,
        default=100,
        help="Number of iterations for numerical derivatives",
    )
    parser.addoption(
        "--deriv-rtol",
        type=float,
        default=1e-4,
        help="Tolerance for numerical derivatives",
    )
    parser.addoption(
        "--deriv-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numerical derivatives",
    )

@pytest.fixture(scope="session")
def deriv_epsilon(request: pytest.FixtureRequest) -> float:
    return request.config.getoption("--deriv-epsilon")

@pytest.fixture(scope="session")
def deriv_rtol(request: pytest.FixtureRequest) -> float:
    return request.config.getoption("--deriv-rtol")

@pytest.fixture(scope="session")
def deriv_atol(request: pytest.FixtureRequest) -> float:
    return request.config.getoption("--deriv-atol")

@pytest.fixture(scope="session")
def deriv_its(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--deriv-its")

@pytest.fixture(scope="session")
def info() -> Info:
    return Info()

@pytest.fixture(scope="session")
def rng() -> Generator:
    return default_rng(seed=42)

@pytest.fixture(scope="session")
def wavelength_array(info: Info) -> FloatVector:
    dv = info.loading.sigma_res
    x0 = info.units.getWavelength(3_000 * Unit("angstrom"))
    x1 = info.units.getWavelength(9_000 * Unit("angstrom"))
    n = floor(log(x1 / x0) / log(1 + dv))
    return x0 * (1 + dv) ** arange(n + 1)

@pytest.fixture(scope="session")
def wavelength_array_linear() -> FloatVector:
    dx = 0.25
    x0 = 3_000.0
    x1 = 9_000.0
    n = floor((x1 - x0) / dx)
    return x0 + dx * arange(n + 1)

###

_this_file: Path = Path(__file__).resolve()

@pytest.fixture(scope="session")
def temp_dir():
    temp = _this_file.parent / "temp"
    temp.mkdir(exist_ok=True)
    yield temp
    rmtree(temp)

### Gaussian

@pytest.fixture(scope="session")
def gaussian_model(info: Info) -> GaussianModel:
    return GaussianModel.create(
        6548.0, info.loading.sigma_res, "n",
        strength=1.0,
        fwhm_v=1000.0,
        v_off=0.0,
        strength_bounds=(0, 1000.0),
        fwhm_v_bounds=(100.0, 10_000.0),
        v_off_bounds=(-5000.0, 5000.0),
        name="test_gaussian",
    )

@pytest.fixture(scope="session")
def v_profile_copy(gaussian_model: GaussianModel) -> VProfileCopy1G:
    return VProfileCopy1G.from_model(
        6584.0,
        "test_gaussian_copy",
        "n",
        gaussian_model,
        strength_scale_value=3.0,
        strength_scale_bounds=(2.0, 4.0),
        strength_scale_fixed=False,
        adapt=True,
    )

### Power law

@pytest.fixture(scope="session")
def powerlaw_model(info: Info) -> PowerLawModel:
    return PowerLawModel.create(
        info.continuum.x0, 
        info.continuum.y0,
        sum(info.continuum.flux_bounds) / 2.0,
        sum(info.continuum.alpha_bounds) / 2.0,
        name="test_powerlaw",
        flux_bounds=info.continuum.flux_bounds,
        alpha_bounds=info.continuum.alpha_bounds,
    )

### Iron

@pytest.fixture(scope="session")
def iron_template(info: Info, wavelength_array: FloatVector) -> IronTemplate:
    return IronTemplate.load_from_cache(
        name="vw2001",
        info=info,
    ).createLogspace(
        sigma_res=info.loading.sigma_res,
        xr=wavelength_array,
    )

@pytest.fixture(scope="session")
def iron_template_cropped(iron_template: IronTemplate) -> IronTemplate:
    temp = iron_template.copy()
    temp.name = temp.name + "_cropped"
    temp.fwhm = iron_template.fwhm[:1].copy()
    temp.data = iron_template.data[:1,:].copy()
    return temp

@pytest.fixture(scope="session")
def iron_model(iron_template: IronTemplate, info: Info) -> IronModel:
    template = iron_template
    model = IronModel.create(
        1.0,
        1000.0,
        scale=info.iron.scale,
        template=template,
        info=info,
        split=template.x[0],
        left=1.0,
        right=1.0,
        allow_interp_fitting=False,
        n_scales=info.convolution.n_scales,
    )
    model.name = "test_iron"
    model.flux.bounds = (0.0, 1000.0)
    model.fwhm.bounds = (template.fwhm[0], 20_000.0)
    return model

### Host galaxy

@pytest.fixture(scope="session")
def host_template(info: Info, wavelength_array: FloatVector) -> HostGalaxyTemplate:
    return HostGalaxyTemplate.load_from_cache(
        name=info.host.sources[0],
        age=info.host.ages[0],
        info=info,
    ).createLogspace(
        sigma_res=info.loading.sigma_res,
        xr=wavelength_array,
    )

@pytest.fixture(scope="session")
def host_template_cropped(host_template: HostGalaxyTemplate) -> HostGalaxyTemplate:
    temp = host_template.copy()
    temp.name = temp.name + "_cropped"
    temp.fwhm = host_template.fwhm[:1].copy()
    temp.data = host_template.data[:1,:].copy()
    return temp

@pytest.fixture(scope="session")
def host_model(host_template: HostGalaxyTemplate, info: Info) -> HostGalaxyModel:
    model = HostGalaxyModel.create(
        1.0,
        1000.0,
        info=info,
        template=host_template,
        allow_interp_fitting=False,
        n_scales=info.convolution.n_scales,
        flux_bounds=info.host.flux_bounds,
        fwhm_bounds=info.host.fwhm_bounds,
        fwhm_fixed=True,
    )
    model.name = "test_host"
    return model

### Balmer

@pytest.fixture(scope="session")
def balmer_continuum_template(info: Info, wavelength_array: FloatVector) -> BalmerContinuumTemplate:
    return BalmerContinuumTemplate.load_from_cache(
        temp=info.balmer.temp,
        tau=info.balmer.tau,
        scale=info.balmer.scale,
        info=info,
    ).createLogspace(
        sigma_res=info.loading.sigma_res,
        xr=wavelength_array,
    )

@pytest.fixture(scope="session")
def balmer_continuum_template_cropped(balmer_continuum_template: BalmerContinuumTemplate) -> BalmerContinuumTemplate:
    temp = balmer_continuum_template.copy()
    temp.name = temp.name + "_cropped"
    temp.fwhm = balmer_continuum_template.fwhm[:1].copy()
    temp.data = balmer_continuum_template.data[:1,:].copy()
    return temp

@pytest.fixture(scope="session")
def balmer_series_template(info: Info, wavelength_array: FloatVector) -> BalmerSeriesTemplate:
    return BalmerSeriesTemplate.load_from_cache(
        name="sh1995",
        temp=info.balmer.temp,
        dens=info.balmer.dens,
        n_u_range=(info.balmer.n_u_min, info.balmer.n_u_max),
        info=info,
    ).createLogspace(
        sigma_res=info.loading.sigma_res,
        xr=wavelength_array,
    )

@pytest.fixture(scope="session")
def balmer_series_template_cropped(balmer_series_template: BalmerSeriesTemplate) -> BalmerSeriesTemplate:
    temp = balmer_series_template.copy()
    temp.name = temp.name + "_cropped"
    temp.fwhm = balmer_series_template.fwhm[:1].copy()
    temp.data = balmer_series_template.data[:1,:].copy()
    return temp

@pytest.fixture(scope="session")
def balmer_model(
    balmer_continuum_template: BalmerContinuumTemplate,
    balmer_series_template: BalmerSeriesTemplate,
    info: Info,
) -> BalmerModel:
    model = BalmerModel.create(
        1.0,
        1000.0,
        1.0,
        edge=info.balmer.edge,
        series_template=balmer_series_template,
        continuum_template=balmer_continuum_template,
        info=info,
        allow_interp_fitting=False,
        n_scales=info.convolution.n_scales,
        flux_bounds=info.balmer.flux_bounds,
        fwhm_bounds=info.balmer.fwhm_bounds,
        ratio_bounds=info.balmer.ratio_bounds,
    )
    model.name = "test_balmer"
    return model