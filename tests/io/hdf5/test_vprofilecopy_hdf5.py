from pathlib import Path

import h5py
import pytest

from quasar_models.io.hdf5.vprofilecopy import (
    hdf5_to_vprofilecopies,
    hdf5_to_vprofilecopy,
    vprofilecopies_to_hdf5,
    vprofilecopy_to_hdf5,
)
from quasar_models.line import VProfileCopy1G


def test_vprofilecopy_hdf5_roundtrip(v_profile_copy: VProfileCopy1G, info, temp_dir: Path):
    """Write a VProfileCopy to HDF5 and read it back, asserting parameters
    and metadata are preserved (within numerical tolerance).
    """
    path = temp_dir / "gaussian_roundtrip.h5"
    # Write
    with h5py.File(path, "w") as f:
        vprofilecopy_to_hdf5(
            v_profile_copy, f, info, 
            group_name="vprofilecopy", 
            force=True,
        )

    # Read and verify
    with h5py.File(path, "r") as f:
        grp = f["vprofilecopy"]
        loaded: VProfileCopy1G = hdf5_to_vprofilecopy(grp, info)

    assert loaded.name == v_profile_copy.name

    # metadata
    assert loaded.wave == pytest.approx(v_profile_copy.wave)
    assert loaded.linetype == v_profile_copy.linetype
    assert loaded.sigma_res == pytest.approx(v_profile_copy.sigma_res)

    # parameters
    for param_name in ["strength_scale", "strength_1", "fwhm_v_1", "v_off_1"]:
        p = getattr(v_profile_copy, param_name)
        q = getattr(loaded, param_name)

        assert q.value == pytest.approx(p.value)
        assert q.bounds == p.bounds
        assert q.fixed == p.fixed


def test_vprofilecopy_hdf5_list_single(v_profile_copy, info, temp_dir: Path):
    """Save and load a list containing a single VProfileCopy."""
    path = temp_dir / "gaussian_list_single.h5"
    models = [v_profile_copy]

    with h5py.File(path, "w") as f:
        vprofilecopies_to_hdf5(
            models, f, info, 
            group_name="vprofilecopy_list",
            force=True,
        )

    with h5py.File(path, "r") as f:
        grp = f["vprofilecopy_list"]
        loaded = hdf5_to_vprofilecopies(grp, info)[0]

    assert v_profile_copy.name == loaded.name
    assert v_profile_copy.wave == pytest.approx(loaded.wave)
    assert v_profile_copy.linetype == loaded.linetype
    assert v_profile_copy.sigma_res == pytest.approx(loaded.sigma_res)

    for param_name in ["strength_scale", "strength_1", "fwhm_v_1", "v_off_1"]:
        p = getattr(v_profile_copy, param_name)
        q = getattr(loaded, param_name)

        assert q.value == pytest.approx(p.value)
        assert q.bounds == p.bounds
        assert q.fixed == p.fixed


def test_vprofilecopy_hdf5_list_multiple(v_profile_copy, info, temp_dir: Path):
    """Save and load multiple VProfileCopy instances."""
    p1 = v_profile_copy
    p2 = v_profile_copy.copy()
    p2.name = "test_vprofilecopy_2"
    p2.strength_scale.value = p2.strength_scale.value * 3.0

    models = [p1, p2]
    path = temp_dir / "gaussian_list_multi.h5"
    with h5py.File(path, "w") as f:
        vprofilecopies_to_hdf5(
            models, f, info, 
            group_name="vprofilecopy_multi", 
            force=True,
        )

    with h5py.File(path, "r") as f:
        grp = f["vprofilecopy_multi"]
        new_models = hdf5_to_vprofilecopies(grp, info)

    for orig, new in zip(models, new_models):
        assert orig.name == new.name
        assert orig.wave == pytest.approx(new.wave)
        assert orig.linetype == new.linetype
        assert orig.sigma_res == pytest.approx(new.sigma_res)

        for param_name in ["strength_scale", "strength_1", "fwhm_v_1", "v_off_1"]:
            p = getattr(orig, param_name)
            q = getattr(new, param_name)

            assert q.value == pytest.approx(p.value)
            assert q.bounds == p.bounds
            assert q.fixed == p.fixed
