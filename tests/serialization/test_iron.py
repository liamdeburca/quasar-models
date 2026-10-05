import json
from pathlib import Path

import numpy as np
from quasar_utils.setup import Info

from quasar_models import IronModel


def test_serialize(iron_model: IronModel, info: Info, temp_dir: Path):
    data = iron_model.serialize(info)
    with open(temp_dir / "iron_model.json", "w") as f:
        json.dump(data, f)

def test_deserialize(iron_model: IronModel, info: Info, temp_dir: Path):
    with open(temp_dir / "iron_model.json", "r") as f:
        data = json.load(f)

    key = next(iter(data))
    cls_name, name = key.split("::")
    assert cls_name == IronModel.__name__

    model = IronModel.deserialize(data[key], name, info)
    assert model.name == iron_model.name
    assert model.n_scales == iron_model.n_scales
    assert model.flux == iron_model.flux
    assert model.fwhm == iron_model.fwhm
    assert np.array_equal(model.template.x, iron_model.template.x)
    assert np.array_equal(model.template.fwhm, iron_model.template.fwhm)
    assert np.allclose(model.template.data, iron_model.template.data)