import h5py
from pathlib import Path

import pytest

from quasar_models.io.hdf5.powerlaw import powerlaw_to_hdf5, hdf5_to_powerlaw


def test_powerlaw_hdf5_roundtrip(powerlaw_model, info, temp_dir: Path):
    """Write a PowerLawModel to HDF5 and read it back, asserting parameters
    and metadata are preserved (within numerical tolerance).
    """
    path = temp_dir / "powerlaw_roundtrip.h5"
    # Write
    with h5py.File(path, "w") as f:
        powerlaw_to_hdf5(powerlaw_model, f, info, group_name="pl", force=True)

    # Read and verify
    with h5py.File(path, "r") as f:
        grp = f["pl"]
        loaded = hdf5_to_powerlaw(grp, info)

    assert loaded.name == powerlaw_model.name
    # x0/y0 are set from `info` when loading; the fixture uses the same info
    assert loaded.x0 == pytest.approx(powerlaw_model.x0)
    assert loaded.y0 == pytest.approx(powerlaw_model.y0)

    # Parameters
    assert loaded.flux.value == pytest.approx(powerlaw_model.flux.value)
    assert loaded.alpha.value == pytest.approx(powerlaw_model.alpha.value)

    # Bounds and fixed flags round-trip
    assert loaded.flux.bounds == powerlaw_model.flux.bounds
    assert loaded.alpha.bounds == powerlaw_model.alpha.bounds
    assert loaded.flux.fixed == powerlaw_model.flux.fixed
    assert loaded.alpha.fixed == powerlaw_model.alpha.fixed
