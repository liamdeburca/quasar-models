import pytest
from numpy import allclose, array, isclose, stack
from numpy.random import Generator
from quasar_typing.numpy import FloatMatrix, FloatVector

from quasar_models.continuum import PowerLawModel
from quasar_models.host import HostGalaxyModel
from quasar_models.line import GaussianModel, VProfileCopy1G
from quasar_models.modeling import PrepareModel, SequentialModel
from quasar_models.modeling.prepare_model import _is_logbinned


def numerical_derivative(
    x: FloatVector,
    model: SequentialModel,
    params: FloatVector,
    bounds: tuple[FloatVector, FloatVector],
    epsilon: float,
) -> FloatMatrix:

    derivs = []
    for i, (p, lb, ub) in enumerate(zip(params, *bounds)):
        params_left = params.copy()
        params_left[i] = max(lb, p - epsilon)
        dp_left = params[i] - params_left[i]
        y_left = model.evaluate(x, params_left)

        params_right = params.copy()
        params_right[i] = min(ub, p + epsilon)
        dp_right = params_right[i] - params[i]
        y_right = model.evaluate(x, params_right)

        dp = dp_left + dp_right
        assert dp > 0.0

        derivs.append((y_right - y_left) / dp)

    return stack(derivs, axis=0)

def test_gaussian(
    gaussian_model: GaussianModel,
    wavelength_array: FloatVector,
    rng: Generator,
    deriv_epsilon: float,
    deriv_its: int,
    deriv_rtol: float,
    deriv_atol: float,
    subtests: pytest.Subtests,
) -> None:
    assert _is_logbinned(wavelength_array)

    gaussian_model.strength.fixed = False
    gaussian_model.fwhm_v.fixed = False
    gaussian_model.v_off.fixed = False

    with PrepareModel(x=wavelength_array, model=gaussian_model) as seq_model:
        m = seq_model._model
        assert m.dx is None
        assert m.kwargs["wave"] == m.wave
        assert m.kwargs["sigma_res"] == m.sigma_res
        assert "dx" not in m.kwargs

        bounds = seq_model.bounds
        for _ in range(deriv_its):
            strength = rng.uniform(*gaussian_model.strength.bounds)
            fwhm_v = rng.uniform(*gaussian_model.fwhm_v.bounds)
            v_off = rng.uniform(*gaussian_model.v_off.bounds)
            params = array([strength, fwhm_v, v_off])
            with subtests.test(msg=f"GaussianModel: {strength=:.1f}, {fwhm_v=:.1f}, {v_off=:.1f}"):
                df_analytical = seq_model.partial_deriv(wavelength_array, params)
                df_numerical = numerical_derivative(wavelength_array, seq_model, params, bounds, deriv_epsilon)
                assert isclose(df_analytical[0], df_numerical[0], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
                assert isclose(df_analytical[1], df_numerical[1], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
                assert isclose(df_analytical[2], df_numerical[2], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0

def test_gaussian_linear(
    gaussian_model: GaussianModel,
    wavelength_array_linear: FloatVector,
    rng: Generator,
    deriv_epsilon: float,
    deriv_its: int,
    deriv_rtol: float,
    deriv_atol: float,
    subtests: pytest.Subtests,
) -> None:
    assert not _is_logbinned(wavelength_array_linear)
    
    gaussian_model.strength.fixed = False
    gaussian_model.fwhm_v.fixed = False
    gaussian_model.v_off.fixed = False

    with PrepareModel(x=wavelength_array_linear, model=gaussian_model) as seq_model:
        m = seq_model._model
        assert m.dx is not None
        assert m.kwargs["wave"] == m.wave
        assert m.kwargs["dx"] == m.dx
        assert "sigma_res" not in m.kwargs

        bounds = seq_model.bounds
        for _ in range(deriv_its):
            strength = rng.uniform(*gaussian_model.strength.bounds)
            fwhm_v = rng.uniform(*gaussian_model.fwhm_v.bounds)
            v_off = rng.uniform(*gaussian_model.v_off.bounds)
            params = array([strength, fwhm_v, v_off])
            with subtests.test(msg=f"GaussianModel: {strength=:.1f}, {fwhm_v=:.1f}, {v_off=:.1f}"):
                df_analytical = seq_model.partial_deriv(wavelength_array_linear, params)
                df_numerical = numerical_derivative(wavelength_array_linear, seq_model, params, bounds, deriv_epsilon)

                assert isclose(df_analytical[0], df_numerical[0], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
                assert isclose(df_analytical[1], df_numerical[1], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
                assert isclose(df_analytical[2], df_numerical[2], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0

def test_sequential(
    gaussian_model: GaussianModel,
    v_profile_copy: VProfileCopy1G,
    wavelength_array: FloatVector,
    rng: Generator,
    deriv_epsilon: float,
    deriv_its: int,
    deriv_rtol: float,
    deriv_atol: float,
    subtests: pytest.Subtests,
) -> None:
    gaussian_model.strength.fixed = False
    gaussian_model.fwhm_v.fixed = False
    gaussian_model.v_off.fixed = False
    v_profile_copy.strength_scale.fixed = False

    assert v_profile_copy.fwhm_v_1.tied.model_name == gaussian_model.name
    assert v_profile_copy.v_off_1.tied.model_name == gaussian_model.name

    mask = (6400.0 < wavelength_array) & (wavelength_array < 6700.0)
    x = wavelength_array[mask]

    with PrepareModel(x=x, model=gaussian_model + v_profile_copy) as seq_model:
        bounds = seq_model.bounds
        for _ in range(deriv_its):
            strength = rng.uniform(*gaussian_model.strength.bounds)
            fwhm_v = rng.uniform(*gaussian_model.fwhm_v.bounds)
            v_off = rng.uniform(*gaussian_model.v_off.bounds)
            strength_scale = rng.uniform(*v_profile_copy.strength_scale.bounds)

            params = array([strength, fwhm_v, v_off, strength_scale])
            with subtests.test(msg=f"GaussianModel: {strength=:.1f}, {fwhm_v=:.1f}, {v_off=:.1f}, {strength_scale=:.1f}"):
                f_analytical = seq_model.evaluate(x, params)
                f_theoretical = gaussian_model.evaluate(x, strength, fwhm_v, v_off) \
                    + v_profile_copy.evaluate(x, strength_scale, strength, fwhm_v, v_off)
                assert allclose(f_analytical, f_theoretical, rtol=deriv_rtol, atol=deriv_atol)

                df_analytical = seq_model.partial_deriv(x, params)[seq_model._free_indices]
                df_numerical = numerical_derivative(x, seq_model, params, bounds, deriv_epsilon)

                assert df_analytical.shape == (4, x.size)
                assert df_numerical.shape == (4, x.size)

                assert allclose(df_analytical[0], df_numerical[0], rtol=deriv_rtol, atol=deriv_atol)
                assert allclose(df_analytical[1], df_numerical[1], rtol=deriv_rtol, atol=deriv_atol)
                # assert allclose(df_analytical[2], df_numerical[2], rtol=deriv_rtol, atol=deriv_atol)
                # assert allclose(df_analytical[3], df_numerical[3], rtol=deriv_rtol, atol=deriv_atol)

def test_powerlaw(
    powerlaw_model: PowerLawModel,
    wavelength_array: FloatVector,
    rng: Generator,
    deriv_epsilon: float,
    deriv_its: int,
    deriv_rtol: float,
    deriv_atol: float,
    subtests: pytest.Subtests,
) -> None:
    powerlaw_model.flux.fixed = False
    powerlaw_model.alpha.fixed = False

    with PrepareModel(x=wavelength_array, model=powerlaw_model) as seq_model:
        for _ in range(deriv_its):
            flux = rng.uniform(*powerlaw_model.flux.bounds)
            alpha = rng.uniform(*powerlaw_model.alpha.bounds)
            params = array([flux, alpha])
            bounds = seq_model.bounds
            with subtests.test(msg=f"PowerLawModel: {flux=:.1f}, {alpha=:.1f}"):
                df_analytical = seq_model.partial_deriv(wavelength_array, params)
                df_numerical = numerical_derivative(wavelength_array, seq_model, params, bounds, deriv_epsilon)
                assert isclose(df_analytical[0], df_numerical[0], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
                assert isclose(df_analytical[1], df_numerical[1], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
    

# def test_iron(
#     iron_model: IronModel,
#     rng: Generator,
#     deriv_epsilon: float,
#     deriv_its: int,
#     deriv_rtol: float,
#     deriv_atol: float,
#     subtests: pytest.Subtests,
# ) -> None:
#     iron_model.flux.fixed = False
#     iron_model.fwhm.fixed = False
#     iron_model.split.fixed = True
#     iron_model.left.value = 1.0
#     iron_model.left.fixed = True
#     iron_model.right.value = 1.0
#     iron_model.right.fixed = True

#     wavelength_array = iron_model.template.x

#     with PrepareModel(x=wavelength_array, model=iron_model) as seq_model:
#         bounds = seq_model.bounds
#         for _ in range(deriv_its):
#             flux = rng.uniform(*iron_model.flux.bounds)
#             fwhm = rng.uniform(*iron_model.fwhm.bounds)
#             params = array([flux, fwhm])
#             with subtests.test(msg=f"IronModel: {flux=:.1f}, {fwhm=:.1f}"):
#                 df_analytical = seq_model.partial_deriv(wavelength_array, params)
#                 df_numerical = numerical_derivative(wavelength_array, seq_model, params, bounds, deriv_epsilon)

#                 assert isclose(df_analytical[0], df_numerical[0], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0
#                 assert isclose(df_analytical[1], df_numerical[1], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0


def test_host(
    host_model: HostGalaxyModel,
    rng: Generator,
    deriv_epsilon: float,
    deriv_its: int,
    deriv_rtol: float,
    deriv_atol: float,
    subtests: pytest.Subtests,
) -> None:
    host_model.flux.fixed = False
    host_model.fwhm.fixed = True

    wavelength_array = host_model.template.x

    with PrepareModel(x=wavelength_array, model=host_model) as seq_model:
        m = seq_model._model
        assert m.evaluate_func.rescaling

        bounds = seq_model.bounds
        for _ in range(deriv_its):
            flux = rng.uniform(*host_model.flux.bounds)
            params = array([flux])
            with subtests.test(msg=f"HostModel: {params=}"):
                df_analytical = seq_model.partial_deriv(wavelength_array, params)
                df_numerical = numerical_derivative(wavelength_array, seq_model, params, bounds, deriv_epsilon)
                assert isclose(df_analytical[0], df_numerical[0], rtol=deriv_rtol, atol=deriv_atol).mean() == 1.0