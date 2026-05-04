import numpy as np
sigma_res: float = 2.3e-4
x = 1400 * (1 + sigma_res)**np.arange(1000)

def test_evaluate() -> None:
    from quasar_models.line.evaluation import evaluate
    y = evaluate(
        x,
        1.0, 0.01, 0.0,
        wave=1549.0, sigma_res=sigma_res, 
    )
    assert x.size == y.size
    assert (y > 0).all()
    assert np.isfinite(y).all()

def test_evaluate_sparse() -> None:
    from quasar_models.line.evaluation import evaluate_sparse
    mask, y = evaluate_sparse(
        x,
        1.0, 0.01, 0.0,
        wave=1549.0, sigma_res=sigma_res, 
    )
    assert mask.size == x.size
    assert mask.any() # Should be some sparse evaluation
    assert (y > 0).all()
    assert np.isfinite(y).all()

def test_fit_deriv() -> None:
    from quasar_models.line.evaluation import fit_deriv
    dy = fit_deriv(
        x,
        1.0, 0.01, 0.0,
        wave=1549.0, sigma_res=sigma_res, 
        fixed={'strength': False, 'sigma_v': False, 'v_off': False},
    )
    assert len(dy) == 3
    assert np.isfinite(dy).all()
    assert x.size == dy[0].size == dy[1].size == dy[2].size
    assert (dy[0] != 0).all()
    assert (dy[1] != 0).all()
    assert (dy[2] != 0).all()

    dy_fixed = fit_deriv(
        x,
        1.0, 0.01, 0.0,
        wave=1549.0, sigma_res=sigma_res,
        fixed={'strength': True, 'sigma_v': True, 'v_off': True},
    )
    assert len(dy_fixed) == 3
    assert np.isfinite(dy_fixed).all()
    assert x.size == dy_fixed[0].size == dy_fixed[1].size == dy_fixed[2].size
    assert (dy_fixed[0] == 0).all()
    assert (dy_fixed[1] == 0).all()
    assert (dy_fixed[2] == 0).all()