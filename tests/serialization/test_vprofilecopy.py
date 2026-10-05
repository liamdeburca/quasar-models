import json
from pathlib import Path

from quasar_utils.setup import Info

from quasar_models.line import VProfileCopy1G


def test_serialize(v_profile_copy: VProfileCopy1G, info: Info, temp_dir: Path):
    data = v_profile_copy.serialize(info)
    with open(temp_dir / "vprofilecopy_model.json", "w") as f:
        json.dump(data, f)


def test_deserialize(v_profile_copy: VProfileCopy1G, info: Info, temp_dir: Path):
    with open(temp_dir / "vprofilecopy_model.json", "r") as f:
        data = json.load(f)

    key = next(iter(data))
    cls_name, name = key.split("::")
    assert cls_name == VProfileCopy1G.__name__

    model = VProfileCopy1G.deserialize(data[key], name, info)
    assert model.name == v_profile_copy.name
    assert model.wave == v_profile_copy.wave
    assert model.linetype == v_profile_copy.linetype
    assert model.n_sigmas == v_profile_copy.n_sigmas
    assert model.strength_scale.value == v_profile_copy.strength_scale.value
    # profile params
    assert model.strength_1.value == v_profile_copy.strength_1.value
    assert model.fwhm_v_1.value == v_profile_copy.fwhm_v_1.value
    assert model.v_off_1.value == v_profile_copy.v_off_1.value
