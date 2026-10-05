from pathlib import Path

import h5py
import pytest

from quasar_models.io.hdf5.gaussian import gaussian_to_hdf5, hdf5_to_gaussian


def test_gaussian_hdf5_roundtrip(gaussian_model, info, temp_dir: Path):
    """Write a GaussianModel to HDF5 and read it back, asserting parameters
    and metadata are preserved (within numerical tolerance).
    """
    path = temp_dir / "gaussian_roundtrip.h5"
    # Write
    with h5py.File(path, "w") as f:
        gaussian_to_hdf5(gaussian_model, f, info, group_name="gauss", force=True)

    # Read and verify
    with h5py.File(path, "r") as f:
        grp = f["gauss"]
        loaded = hdf5_to_gaussian(grp, info)

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
