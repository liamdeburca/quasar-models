from pathlib import Path

from quasar_utils.setup import Info

from quasar_models import PowerLawModel
from quasar_models.io.json import file_to_model, model_to_file
from quasar_models.line import GaussianModel


def test_write_read_json(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    p = Path(temp_dir) / "model.json"
    model_to_file(p, powerlaw_model, info)

    loaded = file_to_model(p, info)
    assert loaded.name == powerlaw_model.name
    assert loaded.x0 == powerlaw_model.x0
    assert loaded.y0 == powerlaw_model.y0
    assert loaded.flux.value == powerlaw_model.flux.value
    assert loaded.alpha.value == powerlaw_model.alpha.value


def test_write_read_yaml(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    p = Path(temp_dir) / "model.yaml"
    model_to_file(p, powerlaw_model, info)

    loaded = file_to_model(p, info)
    assert loaded.name == powerlaw_model.name
    assert loaded.x0 == powerlaw_model.x0
    assert loaded.y0 == powerlaw_model.y0


def test_write_read_toml(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    # TOML support is optional; skip if not installed.
    p = Path(temp_dir) / "model.toml"
    model_to_file(p, powerlaw_model, info)

    loaded = file_to_model(p, info)
    assert loaded.name == powerlaw_model.name
    # x0/y0 are sourced from Info on deserialize, confirm they match
    assert loaded.x0 == info.continuum.x0
    assert loaded.y0 == info.continuum.y0


def test_write_read_toon(powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    # TOML support is optional; skip if not installed.
    p = Path(temp_dir) / "model.toon"
    model_to_file(p, powerlaw_model, info)

    loaded = file_to_model(p, info)
    assert loaded.name == powerlaw_model.name
    # x0/y0 are sourced from Info on deserialize, confirm they match
    assert loaded.x0 == info.continuum.x0
    assert loaded.y0 == info.continuum.y0


def test_compound_json_roundtrip(gaussian_model: GaussianModel, powerlaw_model: PowerLawModel, info: Info, temp_dir: Path):
    # Create simple additive compound
    compound = gaussian_model + powerlaw_model
    p = Path(temp_dir) / "compound.json"
    model_to_file(p, compound, info)

    model = file_to_model(p, info)
    assert model.n_submodels == compound.n_submodels
    origs = sorted((m for m in compound), key=lambda m: m.sorting_key)
    news = sorted((m for m in model), key=lambda m: m.sorting_key)
    for o, n in zip(origs, news):
        assert o.name == n.name
        if hasattr(o, "x0"):
            assert o.x0 == n.x0
        if hasattr(o, "wave"):
            assert o.wave == n.wave
