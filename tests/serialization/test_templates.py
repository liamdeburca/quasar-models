import json
from pathlib import Path

import numpy as np
from quasar_utils.setup import Info

from quasar_models.iron import IronTemplate


def test_iron_template_serialize(iron_template: IronTemplate, info: Info, temp_dir: Path):
    data = iron_template.serialize(info)
    with open(temp_dir / "iron_template.json", "w") as f:
        json.dump(data, f)

def test_iron_template_deserialize(iron_template: IronTemplate, info: Info,temp_dir: Path):
    with open(temp_dir / "iron_template.json", "r") as f:
        data = json.load(f)

    template = IronTemplate.deserialize(data, info)
    assert np.array_equal(template.x, iron_template.x)
    assert np.array_equal(template.fwhm, iron_template.fwhm)
    assert np.allclose(template.data, iron_template.data)
    assert template.x_norm == iron_template.x_norm
    assert template.fwhm_norm == iron_template.fwhm_norm
    assert template.normalisation == iron_template.normalisation
    assert template.name == iron_template.name
    assert template.path == iron_template.path
    assert template.is_logspace == iron_template.is_logspace
    assert template.sigma_res == iron_template.sigma_res
    

