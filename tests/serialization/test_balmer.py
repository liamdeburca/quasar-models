import json
from pathlib import Path

import numpy as np
from quasar_utils.setup import Info

from quasar_models import BalmerModel


def test_serialize(balmer_model: BalmerModel, info: Info, temp_dir: Path):
    data = balmer_model.serialize(info)
    with open(temp_dir / "balmer_model.json", "w") as f:
        json.dump(data, f)

def test_deserialize(balmer_model: BalmerModel, info: Info, temp_dir: Path):
    with open(temp_dir / "balmer_model.json", "r") as f:
        data = json.load(f)

    key = next(iter(data))
    cls_name, name = key.split("::")
    assert cls_name == BalmerModel.__name__

    model = BalmerModel.deserialize(data[key], name, info)
    assert model.name == balmer_model.name
    assert model.flux == balmer_model.flux
    assert model.fwhm == balmer_model.fwhm
    assert model.ratio == balmer_model.ratio
    assert np.array_equal(model.continuum_template.x, balmer_model.continuum_template.x)
    assert np.array_equal(model.continuum_template.fwhm, balmer_model.continuum_template.fwhm)
    assert np.allclose(model.continuum_template.data, balmer_model.continuum_template.data)
    assert np.array_equal(model.series_template.x, balmer_model.series_template.x)
    assert np.array_equal(model.series_template.fwhm, balmer_model.series_template.fwhm)
    assert np.allclose(model.series_template.data, balmer_model.series_template.data)