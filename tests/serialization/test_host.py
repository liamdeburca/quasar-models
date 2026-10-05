import json
from pathlib import Path

import numpy as np
from quasar_utils.setup import Info

from quasar_models import HostGalaxyModel


def test_serialize(host_model: HostGalaxyModel, info: Info, temp_dir: Path):
    data = host_model.serialize(info)
    with open(temp_dir / "host_model.json", "w") as f:
        json.dump(data, f)

def test_deserialize(host_model: HostGalaxyModel, info: Info, temp_dir: Path):
    with open(temp_dir / "host_model.json", "r") as f:
        data = json.load(f)

    key = next(iter(data))
    cls_name, name = key.split("::")
    assert cls_name == HostGalaxyModel.__name__

    model = HostGalaxyModel.deserialize(data[key], name, info)
    assert model.name == host_model.name
    assert model.n_scales == host_model.n_scales
    assert model.flux == host_model.flux
    assert model.fwhm == host_model.fwhm
    assert np.array_equal(model.template.x, host_model.template.x)
    assert np.array_equal(model.template.fwhm, host_model.template.fwhm)
    assert np.allclose(model.template.data, host_model.template.data)