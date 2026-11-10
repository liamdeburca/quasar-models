import numpy as np
import pytest

N: int = 1000


@pytest.fixture
def fwhm_array() -> np.ndarray:
    return (1000 + 250 * np.arange(81)).astype(float)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def get_random_fwhm(rng: np.random.Generator, fwhm_array: np.ndarray) -> float:
    return rng.uniform(fwhm_array[0], fwhm_array[-1])


def test_identify_closest_idx(
    subtests: pytest.Subtests,
    rng: np.random.Generator,
    fwhm_array: np.ndarray,
):
    from quasar_models._core.convolution import identify_closest_idx

    for i, fwhm in enumerate(fwhm_array):
        with subtests.test(i=i, fwhm=fwhm):
            idx = identify_closest_idx(fwhm_array, fwhm)
            assert idx == i

    for i in range(N):
        with subtests.test(i=i):
            fwhm = get_random_fwhm(rng, fwhm_array)
            idx = identify_closest_idx(fwhm_array, fwhm)

            assert 0 <= idx
            assert idx <= len(fwhm_array) - 1

            fwhm_left = fwhm_array[idx]
            assert fwhm_left <= fwhm

            if idx != len(fwhm_array) - 1:
                fwhm_right = fwhm_array[idx + 1]
                assert fwhm < fwhm_right


def test_identify_closest_idx_for_deriv(
    subtests: pytest.Subtests,
    rng: np.random.Generator,
    fwhm_array: np.ndarray,
):
    from quasar_models._core.convolution import identify_closest_idx_for_deriv

    for i, fwhm in enumerate(fwhm_array):
        with subtests.test(i=i, fwhm=fwhm):
            idx = identify_closest_idx_for_deriv(fwhm_array, fwhm)
            assert idx == max(i - 1, 0)

    for i in range(N):
        with subtests.test(i=i):
            fwhm = get_random_fwhm(rng, fwhm_array)
            idx = identify_closest_idx_for_deriv(fwhm_array, fwhm)

            assert 0 <= idx
            assert idx <= len(fwhm_array) - 1

            if idx != 0:
                fwhm_left = fwhm_array[idx]
                assert fwhm_left <= fwhm

            if idx != len(fwhm_array) - 1:
                fwhm_right = fwhm_array[idx + 1]
                assert fwhm < fwhm_right