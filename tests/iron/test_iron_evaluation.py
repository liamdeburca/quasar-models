import numpy as np
from quasar_models.iron import IronModel, IronTemplate
from quasar_utils.setup import Info

info = Info()
sigma_res: float = info.loading['sigma_res']
scale: float = info.iron['scale']
x = 1000 * (1 + sigma_res)**np.arange(4_000)

def test_adapt_iron_template() -> None:
    _template = IronTemplate.load('vw_2001', info)
    template = _template.createLogspace(xr=x, inplace=False, keep_x=True)
    assert np.array_equal(template.x, x)

def test_evaluate() -> None:
    from quasar_models.iron.evaluation import evaluate
    _template = IronTemplate.load('vw_2001', info)
    template = _template.createLogspace(xr=x, inplace=False, keep_x=True)

    y = evaluate(
        x,
        1.0, template.fwhm[0], 0.0, 1.0, 1.0, 
        sigma_res=sigma_res, scale=scale, template=template,
    )
    assert x.size == y.size
    assert np.isclose(template.data[0], y).all()

    y = evaluate(
        x,
        1.0, template.fwhm[-1], 0.0, 1.0, 1.0, 
        sigma_res=sigma_res, scale=scale, template=template,
    )
    assert x.size == y.size
    assert np.isclose(template.data[-1], y).all()

    y = evaluate(
        x,
        1.0, template.fwhm[:2].mean(), 2250, 0, 1.0,
        sigma_res=sigma_res, scale=scale, template=template,
    )
    assert x.size == y.size

def test_evaluate_with_interpolation() -> None:
    from quasar_models.iron.evaluation import evaluate
    _template = IronTemplate.load('vw_2001', info)
    template = _template.createLogspace(xr=x, inplace=False, keep_x=True)

    from quasar_utils.interpolation import create_interp_matrix
    _x = 1000 * (1 + np.pi * sigma_res)**np.arange(2_000)

    y = evaluate(
        _x,
        1.0, template.fwhm[0], 0.0, 1.0, 1.0, 
        sigma_res=sigma_res, scale=scale, template=template,
        interpolation_matrix=create_interp_matrix(template.x, _x)
    )
    assert _x.size == y.size
    assert np.isclose(
        y, 
        np.interp(_x, template.x, template.data[0], left=0, right=0),
    ).all()

def test_evaluate_interp() -> None:
    from quasar_models.iron.evaluation import evaluate_interp
    _template = IronTemplate.load('vw_2001', info)
    template = _template.createLogspace(xr=x, inplace=False, keep_x=True)

    y = evaluate_interp(
        x,
        1.0, template.fwhm[0],
        template=template,
    )
    assert x.size == y.size
    assert np.isclose(template.data[0], y).all()

    y = evaluate_interp(
        x,
        1.0, template.fwhm[:2].mean(),
        template=template,
    )
    assert x.size == y.size
    assert np.isfinite(y).all()

def test_evaluate_sparse() -> None:
    from quasar_models.iron.evaluation import evaluate_sparse
    _template = IronTemplate.load('vw_2001', info)
    template = _template.createLogspace(xr=x, inplace=False, keep_x=True)

    mask, y = evaluate_sparse(
        x,
        1.0, template.fwhm[0], 2250, 0, 1.0, 
        sigma_res=sigma_res, scale=scale, template=template,
    )
    assert mask.size == y.size == x.size
    assert mask.all()

def test_fit_deriv() -> None:
    from quasar_models.iron.evaluation import fit_deriv
    _template = IronTemplate.load('vw_2001', info)
    template = _template.createLogspace(xr=x, inplace=False, keep_x=True)

    dy = fit_deriv(
        x,
        1.0, template.fwhm[:2].mean(), 2250, 0, 1.0,
        sigma_res=sigma_res, scale=scale, template=template,
        fixed=dict(flux=False, fwhm=False, split=False, left=False, right=False),
    )
    assert all(_dy.size == x.size for _dy in dy)
    assert not all((_dy == 0).all() for _dy in dy)

    dy = fit_deriv(
        x,
        1.0, template.fwhm[:2].mean(), 2250, 0, 1.0,
        sigma_res=sigma_res, scale=scale, template=template,
        fixed=dict(flux=True, fwhm=True, split=True, left=True, right=True)
    )
    assert all(_dy.size == x.size for _dy in dy)
    assert all((_dy == 0).all() for _dy in dy)
