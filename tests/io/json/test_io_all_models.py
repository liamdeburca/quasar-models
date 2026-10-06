from pathlib import Path

import pytest
from quasar_utils.setup import Info

from quasar_models.io.json import file_to_model, model_to_file

FORMATS = [".json", ".yaml", ".toml", ".toon"]


def _compare_models(orig, loaded):
    assert loaded.name == orig.name
    # Common optional attributes
    if hasattr(orig, "x0"):
        assert loaded.x0 == orig.x0
    if hasattr(orig, "y0"):
        assert loaded.y0 == orig.y0
    if hasattr(orig, "wave"):
        assert loaded.wave == orig.wave

    # Common parameter names
    for attr in ("flux", "alpha", "strength", "fwhm", "v_off"):
        if hasattr(orig, attr) and hasattr(loaded, attr):
            a = getattr(orig, attr)
            b = getattr(loaded, attr)
            # astropy Parameters expose .value
            if hasattr(a, "value") and hasattr(b, "value"):
                assert a.value == b.value


def _roundtrip_and_check(subtests: pytest.Subtests, model, info: Info, temp_dir: Path):
    for ext in FORMATS:
        p = Path(temp_dir) / f"{model.__class__.__name__}{ext}"
        model_to_file(p, model, info)
        assert p.exists()

        size = p.stat().st_size
        assert size > 0
        with subtests.test(msg=f"{ext=}: {size=}"):
            pass

        loaded = file_to_model(p, info)
        _compare_models(model, loaded)


def test_powerlaw_roundtrips(subtests: pytest.Subtests, powerlaw_model, info: Info, temp_dir: Path):
    _roundtrip_and_check(subtests, powerlaw_model, info, temp_dir)


def test_gaussian_roundtrips(subtests: pytest.Subtests, gaussian_model, info: Info, temp_dir: Path):
    _roundtrip_and_check(subtests, gaussian_model, info, temp_dir)


def test_vprofilecopy_roundtrips(subtests: pytest.Subtests, v_profile_copy, info: Info, temp_dir: Path):
    _roundtrip_and_check(subtests, v_profile_copy, info, temp_dir)


def test_iron_roundtrips(subtests: pytest.Subtests, iron_model, info: Info, temp_dir: Path):
    _roundtrip_and_check(subtests, iron_model, info, temp_dir)


def test_host_roundtrips(subtests: pytest.Subtests, host_model, info: Info, temp_dir: Path):
    _roundtrip_and_check(subtests, host_model, info, temp_dir)


def test_balmer_roundtrips(subtests: pytest.Subtests, balmer_model, info: Info, temp_dir: Path):
    _roundtrip_and_check(subtests, balmer_model, info, temp_dir)
