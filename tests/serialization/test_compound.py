import json
from pathlib import Path

from quasar_utils.setup import Info

from quasar_models import PowerLawModel
from quasar_models.line import GaussianModel
from quasar_models.io.serialization.compound import (
    serialize_compound_model,
    deserialize_compound_model,
)


def test_serialize_single_model(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    data = serialize_compound_model(powerlaw_model, info)
    with open(temp_dir / "compound_single.json", "w") as f:
        json.dump(data, f)

    # Round-trip
    with open(temp_dir / "compound_single.json", "r") as f:
        loaded = json.load(f)

    model = deserialize_compound_model(loaded, info)
    assert model.name == powerlaw_model.name
    assert model.x0 == powerlaw_model.x0
    assert model.y0 == powerlaw_model.y0


def test_serialize_compound(gaussian_model: GaussianModel, powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    # Create a simple additive compound M = Gaussian + PowerLaw
    g = gaussian_model
    p = powerlaw_model
    compound = g + p

    data = serialize_compound_model(compound, info)
    with open(temp_dir / "compound_multi.json", "w") as f:
        json.dump(data, f)

    with open(temp_dir / "compound_multi.json", "r") as f:
        loaded = json.load(f)

    model = deserialize_compound_model(loaded, info)

    # Compound should have same number of submodels
    assert model.n_submodels == compound.n_submodels
    # Check sorted order and parameters
    origs = sorted((m for m in compound), key=lambda m: m.sorting_key)
    news = sorted((m for m in model), key=lambda m: m.sorting_key)
    for o, n in zip(origs, news):
        assert o.name == n.name
        # Rough check for a couple of attributes depending on type
        if hasattr(o, "x0"):
            assert o.x0 == n.x0
        if hasattr(o, "wave"):
            assert o.wave == n.wave
