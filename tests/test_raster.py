import numpy as np
import pytest
from quasar_utils.raster import rasterise
from quasar_utils.setup import Info

from quasar_models.iron import IronModel, IronTemplate
from quasar_models.modeling import PrepareModel


@pytest.fixture
def info() -> Info:
    info = Info()
    info.iron.flux_bounds = (1.0, 10.0)
    return info

@pytest.fixture
def template(info: Info) -> IronTemplate:
    return IronTemplate\
        .load(path="vw2001", info=info)\
        .createLogspace(sigma_res=info.loading.sigma_res)

@pytest.fixture
def model(
    info: Info,
    template: IronTemplate,
) -> IronModel: 
    model = IronModel.create(
        info.iron.flux_bounds[0],
        info.iron.fwhm_bounds[0],
        scale=info.iron.scale,
        template=template,
    )
    model.flux.bounds = info.iron.flux_bounds
    model.fwhm.bounds = info.iron.fwhm_bounds
    return model

def test_main(
    subtests: pytest.Subtests,
    template: IronTemplate,
    info: Info,
    model: IronModel,
) -> None:

    x = template.x
    dy = np.ones_like(x, dtype=np.float64)

    N_tests: int = 1000
    for _ in range(N_tests):
        flux = np.random.uniform(*model.flux.bounds)
        fwhm = np.random.uniform(*model.fwhm.bounds)

        fwhm_idx = np.searchsorted(template.fwhm, fwhm, side="right")
        fwhm_lb = template.fwhm[fwhm_idx - 1]
        fwhm_ub = template.fwhm[fwhm_idx]

        model.flux.value = flux
        model.fwhm.value = fwhm
        with PrepareModel(x=template.x, model=model) as _model:
            y = _model(template.x)

        chi2s = rasterise.__wrapped__(
            y, 
            dy, 
            template.fwhm, 
            template.data / template.normalisation,
            flux_bounds=model.flux.bounds,
            fwhm_bounds=model.fwhm.bounds,
        )[0]
        sol_idx = np.argmin(chi2s)
        chi2_sol = chi2s[sol_idx]
        fwhm_sol = template.fwhm[sol_idx]
        _fwhm_sol = info.units.formatC(fwhm_sol, with_unit=False)

        with subtests.test(
            flux=info.units.formatFlux(flux, with_unit=False),
            fwhm=info.units.formatC(fwhm, with_unit=False),
            fwhm_lb=info.units.formatC(fwhm_lb, with_unit=False),
            fwhm_ub=info.units.formatC(fwhm_ub, with_unit=False),
            chi2_sol=round(chi2_sol, 1),
            fwhm_sol=_fwhm_sol,
        ):
            assert fwhm_sol in (fwhm_lb, fwhm_ub)
