import json
from pathlib import Path

from quasar_utils.setup import Info

from quasar_models import GaussianModel


def test_serialize(gaussian_model: GaussianModel, info: Info, temp_dir: Path):
    data = gaussian_model.serialize(info)
    with open(temp_dir / "gaussian_model.json", "w") as f:
        json.dump(data, f)

def test_deserialize(gaussian_model: GaussianModel, info: Info, temp_dir: Path):
    with open(temp_dir / "gaussian_model.json", "r") as f:
        data = json.load(f)

    key = next(iter(data))
    cls_name, name = key.split("::")
    assert cls_name == GaussianModel.__name__

    model = GaussianModel.deserialize(data[key], name, info)
    assert model.name == gaussian_model.name
    assert model.wave == gaussian_model.wave
    assert model.linetype == gaussian_model.linetype
    assert model.n_sigmas == gaussian_model.n_sigmas
    assert model.strength.value == gaussian_model.strength.value
    assert model.fwhm_v.value == gaussian_model.fwhm_v.value
    assert model.v_off.value == gaussian_model.v_off.value
