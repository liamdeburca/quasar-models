"""
Tests whether the iron, balmer (series + continuum), and host templates can be 
instantiated. 
"""

def test_can_instantiate_iron_templates():
    from quasar_models.iron.setup import main
    main()


def test_can_instantiate_balmer_continuum_templates():
    from quasar_models.balmer.continuum.setup import main
    main()


def test_can_instantiate_balmer_series_templates():
    from quasar_models.balmer.series.setup import main
    main()


def test_can_instantiate_host_templates():
    from quasar_models.host.setup import main
    main()