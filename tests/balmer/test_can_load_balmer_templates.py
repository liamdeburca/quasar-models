import numpy as np

from pathlib import Path
_this_file: Path = Path(__file__).resolve()

from quasar_utils.setup import Info
_info = Info()

def test_path_to_cache() -> None:
    from quasar_models.balmer.balmer_template import PATH_TO_CACHE
    assert PATH_TO_CACHE == _this_file.parents[2] / '.cache/balmer_templates'
    assert PATH_TO_CACHE.exists()

def test_templates_exist() -> None:
    from quasar_models.balmer.balmer_template import PATH_TO_CACHE
    assert (PATH_TO_CACHE / 'qsfit_01.fits').exists()
    assert (PATH_TO_CACHE / 'qsfit_03.fits').exists()
    assert (PATH_TO_CACHE / 'qsfit_05.fits').exists()
    assert (PATH_TO_CACHE / 'qsfit_07.fits').exists()
    assert (PATH_TO_CACHE / 'qsfit_10.fits').exists()

def test_can_load_template() -> None:
    from quasar_models.balmer import BalmerTemplate
    _ = BalmerTemplate.load('qsfit_01', _info)

def test_can_save_and_load_template() -> None:
    from quasar_models.balmer import BalmerTemplate
    from os import remove

    template = BalmerTemplate.load('qsfit_01', _info)
    temp_path = template.save('test.fits', overwrite=True)
    loaded_template = BalmerTemplate.load(temp_path, _info)

    try:
        assert np.isclose(template.fwhm, loaded_template.fwhm).all()
        assert np.isclose(template.x, loaded_template.x).all()
        assert np.isclose(template.data, loaded_template.data).all()
    except Exception as e:
        raise e
    finally:
        remove(temp_path)