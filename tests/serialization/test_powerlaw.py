import json
from pathlib import Path

from quasar_utils.setup import Info

from quasar_models import PowerLawModel


def test_serialize(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    data = powerlaw_model.serialize(info)
    with open(temp_dir / "powerlaw_model.json", "w") as f:
        json.dump(data, f)

def test_deserialize(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    with open(temp_dir / "powerlaw_model.json", "r") as f:
        data = json.load(f)

    key = next(iter(data))
    cls_name, name = key.split("::")
    assert cls_name == PowerLawModel.__name__

    model = PowerLawModel.deserialize(data[key], name, info)
    assert model.name == powerlaw_model.name
    assert model.x0 == powerlaw_model.x0
    assert model.y0 == powerlaw_model.y0
    assert model.flux.value == powerlaw_model.flux.value
    assert model.alpha.value == powerlaw_model.alpha.value
