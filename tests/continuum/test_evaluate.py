import numpy as np

x = np.linspace(1000, 4000, 10, dtype=float)


def test_func() -> None:
    from quasar_core.modelling.powerlaw import evaluate

    y = evaluate(x, 1.0, -1.0, x0=1450.0)
    assert x.size == y.size
    assert (y > 0).all()
    assert np.isfinite(y).all()


def test_deriv() -> None:
    from quasar_core.modelling.powerlaw import fit_deriv_all, choose_fit_deriv_func

    dy = fit_deriv_all(x, 1.0, -1.0, x0=1450.0)

    assert len(dy) == 2
    assert np.isfinite(dy).all()
    assert x.size == dy[0].size == dy[1].size
    assert (dy[0] != 0).all()
    assert (dy[1] != 0).all()

    fit_deriv_func = choose_fit_deriv_func(fixed={"flux": True, "alpha": True})

    dy_fixed = fit_deriv_func(x, 1.0, -1.0, x0=1450.0)
    assert len(dy_fixed) == 2
    assert np.isfinite(dy_fixed).all()
    assert x.size == dy_fixed[0].size == dy_fixed[1].size
    assert (dy_fixed[0] == 0).all()
    assert (dy_fixed[1] == 0).all()
