from pathlib import Path

import h5py
import pytest

from quasar_models import PowerLawModel
from quasar_models.io.hdf5.powerlaw import (
    hdf5_to_powerlaw,
    hdf5_to_powerlaws,
    powerlaw_to_hdf5,
    powerlaws_to_hdf5,
)


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


def test_powerlaw_hdf5_list_single(powerlaw_model, info, temp_dir: Path):
    """Save and load a list containing a single PowerLawModel."""
    path = temp_dir / "powerlaw_list_single.h5"
    models = [powerlaw_model]

    with h5py.File(path, "w") as f:
        powerlaws_to_hdf5(
            models, f, info,
            group_name="pl_list",
            force=True,
        )

    with h5py.File(path, "r") as f:
        grp = f["pl_list"]
        # create blank models to populate
        new_models = hdf5_to_powerlaws(grp, info)

    nm = new_models[0]
    pm = powerlaw_model
    assert nm.name == pm.name
    assert nm.x0 == pytest.approx(pm.x0)
    assert nm.y0 == pytest.approx(pm.y0)
    assert nm.flux.value == pytest.approx(pm.flux.value)
    assert nm.alpha.value == pytest.approx(pm.alpha.value)


def test_powerlaw_hdf5_list_multiple(powerlaw_model, info, temp_dir: Path):
    """Save and load a list containing multiple PowerLawModel instances."""
    # create a second model with different params
    m1 = powerlaw_model
    m2 = powerlaw_model.copy()
    m2.name = "test_powerlaw_2"
    m2.flux.value = m2.flux.value * 2.0
    m2.alpha.value = m2.alpha.value + 0.1

    models = [m1, m2]
    path = temp_dir / "powerlaw_list_multi.h5"
    with h5py.File(path, "w") as f:
        powerlaws_to_hdf5(
            models, f, info,
            group_name="pl_multi",
            force=True,
        )

    with h5py.File(path, "r") as f:
        grp = f["pl_multi"]
        new_models = [PowerLawModel() for _ in range(2)]
        new_models = hdf5_to_powerlaws(grp, info)

    for orig, new in zip(models, new_models):
        assert new.name == orig.name
        assert new.x0 == pytest.approx(orig.x0)
        assert new.y0 == pytest.approx(orig.y0)
        assert new.flux.value == pytest.approx(orig.flux.value)
        assert new.alpha.value == pytest.approx(orig.alpha.value)
