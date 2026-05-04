import numpy as np

from pathlib import Path
_this_file: Path = Path(__file__).resolve()

from quasar_utils.setup import Info
_info = Info()

def test_path_to_cache() -> None:
    from quasar_models.iron.iron_template import PATH_TO_CACHE
    assert PATH_TO_CACHE == _this_file.parents[2] / '.cache/iron_templates'
    assert PATH_TO_CACHE.exists()

def test_templates_exist() -> None:
    from quasar_models.iron.iron_template import PATH_TO_CACHE
    assert (PATH_TO_CACHE / 'vw_2001.fits').exists()
    assert (PATH_TO_CACHE / 'v_2003.fits').exists()
    assert (PATH_TO_CACHE / 'bw.fits').exists()

def test_can_load_vw_2001() -> None:
    from quasar_models.iron import IronTemplate
    _ = IronTemplate.load('vw_2001', _info)

def test_can_load_v_2003() -> None:
    from quasar_models.iron import IronTemplate
    _ = IronTemplate.load('v_2003', _info)

def test_can_load_bw() -> None:
    from quasar_models.iron import IronTemplate
    _ = IronTemplate.load('bw', _info)

def test_can_save_and_load_template() -> None:
    from quasar_models.iron import IronTemplate
    from os import remove

    template = IronTemplate.load('vw_2001', _info)
    temp_path = template.save('test.fits', overwrite=True)
    loaded_template = IronTemplate.load(temp_path, _info)

    try:
        assert np.isclose(template.fwhm, loaded_template.fwhm).all()
        assert np.isclose(template.x, loaded_template.x).all()
        assert np.isclose(template.data, loaded_template.data).all()
    except Exception as e:
        raise e
    finally:
        remove(temp_path)