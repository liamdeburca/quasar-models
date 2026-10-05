from pathlib import Path

import h5py
import pytest

from quasar_models.io.hdf5.gaussian import (
    gaussian_to_hdf5,
    gaussians_to_hdf5,
    hdf5_to_gaussian,
    hdf5_to_gaussians,
)
from quasar_models.line import GaussianModel


def test_gaussian_hdf5_roundtrip(gaussian_model, info, temp_dir: Path):
    """Write a GaussianModel to HDF5 and read it back, asserting parameters
    and metadata are preserved (within numerical tolerance).
    """
    path = temp_dir / "gaussian_roundtrip.h5"
    # Write
    with h5py.File(path, "w") as f:
        gaussian_to_hdf5(
            gaussian_model, 
            f, 
            info, 
            group_name="gauss", 
            force=True,
        )

    # Read and verify
    with h5py.File(path, "r") as f:
        grp = f["gauss"]
        loaded = hdf5_to_gaussian(
            grp, 
            info,
        )

    assert loaded.name == gaussian_model.name

    # metadata
    assert loaded.wave == pytest.approx(gaussian_model.wave)
    assert loaded.linetype == gaussian_model.linetype
    assert loaded.sigma_res == pytest.approx(gaussian_model.sigma_res)

    # parameters
    assert loaded.strength.value == pytest.approx(gaussian_model.strength.value)
    assert loaded.fwhm_v.value == pytest.approx(gaussian_model.fwhm_v.value)
    assert loaded.v_off.value == pytest.approx(gaussian_model.v_off.value)

    # bounds and fixed flags
    assert loaded.strength.bounds == gaussian_model.strength.bounds
    assert loaded.fwhm_v.bounds == gaussian_model.fwhm_v.bounds
    assert loaded.v_off.bounds == gaussian_model.v_off.bounds

    assert loaded.strength.fixed == gaussian_model.strength.fixed
    assert loaded.fwhm_v.fixed == gaussian_model.fwhm_v.fixed
    assert loaded.v_off.fixed == gaussian_model.v_off.fixed


def test_gaussian_hdf5_list_single(gaussian_model, info, temp_dir: Path):
    """Save and load a list containing a single GaussianModel."""
    path = temp_dir / "gaussian_list_single.h5"
    models = [gaussian_model]

    with h5py.File(path, "w") as f:
        gaussians_to_hdf5(
            models, f, info, 
            group_name="gauss_list",
            force=True,
        )

    with h5py.File(path, "r") as f:
        grp = f["gauss_list"]
        new_models = [GaussianModel() for _ in range(1)]
        new_models = hdf5_to_gaussians(grp, info)

    nm = new_models[0]
    gm = gaussian_model
    assert nm.name == gm.name
    assert nm.wave == pytest.approx(gm.wave)
    assert nm.linetype == gm.linetype
    assert nm.sigma_res == pytest.approx(gm.sigma_res)
    assert nm.strength.value == pytest.approx(gm.strength.value)
    assert nm.fwhm_v.value == pytest.approx(gm.fwhm_v.value)
    assert nm.v_off.value == pytest.approx(gm.v_off.value)


def test_gaussian_hdf5_list_multiple(gaussian_model, info, temp_dir: Path):
    """Save and load multiple GaussianModel instances."""
    m1 = gaussian_model
    m2 = gaussian_model.copy()
    m2.name = "test_gaussian_2"
    m2.strength.value = m2.strength.value * 3.0
    m2.fwhm_v.value = m2.fwhm_v.value + 100.0

    models = [m1, m2]
    path = temp_dir / "gaussian_list_multi.h5"
    with h5py.File(path, "w") as f:
        gaussians_to_hdf5(
            models, f, info, 
            group_name="gauss_multi", 
            force=True,
        )

    with h5py.File(path, "r") as f:
        grp = f["gauss_multi"]
        new_models = [GaussianModel() for _ in range(2)]
        new_models = hdf5_to_gaussians(grp, info)

    for orig, new in zip(models, new_models):
        assert new.name == orig.name
        assert new.wave == pytest.approx(orig.wave)
        assert new.linetype == orig.linetype
        assert new.sigma_res == pytest.approx(orig.sigma_res)
        assert new.strength.value == pytest.approx(orig.strength.value)
        assert new.fwhm_v.value == pytest.approx(orig.fwhm_v.value)
        assert new.v_off.value == pytest.approx(orig.v_off.value)
