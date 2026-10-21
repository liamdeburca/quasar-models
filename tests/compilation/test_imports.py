"""
Test module: Verify that all public APIs can be imported from quasar_models.

This test suite systematically tests the import of every class and function
exported via __all__ from all major quasar_models modules.
"""



class TestRootModuleImports:
    """Test imports from the root quasar_models module."""

    def test_import_balmer_model(self):
        """Verify BalmerModel can be imported from quasar_models."""
        from quasar_models import BalmerModel
        assert BalmerModel is not None

    def test_import_gaussian_model(self):
        """Verify GaussianModel can be imported from quasar_models."""
        from quasar_models import GaussianModel
        assert GaussianModel is not None

    def test_import_host_galaxy_model(self):
        """Verify HostGalaxyModel can be imported from quasar_models."""
        from quasar_models import HostGalaxyModel
        assert HostGalaxyModel is not None

    def test_import_iron_model(self):
        """Verify IronModel can be imported from quasar_models."""
        from quasar_models import IronModel
        assert IronModel is not None

    def test_import_powerlaw_model(self):
        """Verify PowerLawModel can be imported from quasar_models."""
        from quasar_models import PowerLawModel
        assert PowerLawModel is not None

    def test_import_vprofilecopy(self):
        """Verify VProfileCopy can be imported from quasar_models."""
        from quasar_models import VProfileCopy
        assert VProfileCopy is not None


class TestBalmerModuleImports:
    """Test imports from quasar_models.balmer submodule."""

    def test_import_balmer_model(self):
        """Verify BalmerModel can be imported from quasar_models.balmer."""
        from quasar_models.balmer import BalmerModel
        assert BalmerModel is not None

    def test_import_balmer_continuum_template(self):
        """Verify BalmerContinuumTemplate can be imported."""
        from quasar_models.balmer import BalmerContinuumTemplate
        assert BalmerContinuumTemplate is not None

    def test_import_balmer_series_template(self):
        """Verify BalmerSeriesTemplate can be imported."""
        from quasar_models.balmer import BalmerSeriesTemplate
        assert BalmerSeriesTemplate is not None


class TestBalmerContinuumModuleImports:
    """Test imports from quasar_models.balmer.continuum submodule."""

    def test_import_balmer_continuum_template(self):
        """Verify BalmerContinuumTemplate can be imported."""
        from quasar_models.balmer.continuum import BalmerContinuumTemplate
        assert BalmerContinuumTemplate is not None

    def test_import_path_to_cache(self):
        """Verify PATH_TO_CACHE can be imported from continuum."""
        from quasar_models.balmer.continuum import PATH_TO_CACHE
        assert PATH_TO_CACHE is not None


class TestBalmerSeriesModuleImports:
    """Test imports from quasar_models.balmer.series submodule."""

    def test_import_balmer_series_template(self):
        """Verify BalmerSeriesTemplate can be imported."""
        from quasar_models.balmer.series import BalmerSeriesTemplate
        assert BalmerSeriesTemplate is not None

    def test_import_path_to_cache(self):
        """Verify PATH_TO_CACHE can be imported from series."""
        from quasar_models.balmer.series import PATH_TO_CACHE
        assert PATH_TO_CACHE is not None

    def test_import_path_to_data(self):
        """Verify PATH_TO_DATA can be imported from series."""
        from quasar_models.balmer.series import PATH_TO_DATA
        assert PATH_TO_DATA is not None


class TestContinuumModuleImports:
    """Test imports from quasar_models.continuum submodule."""

    def test_import_powerlaw_model(self):
        """Verify PowerLawModel can be imported from continuum."""
        from quasar_models.continuum import PowerLawModel
        assert PowerLawModel is not None


class TestHostModuleImports:
    """Test imports from quasar_models.host submodule."""

    def test_import_host_galaxy_model(self):
        """Verify HostGalaxyModel can be imported from host."""
        from quasar_models.host import HostGalaxyModel
        assert HostGalaxyModel is not None

    def test_import_host_galaxy_template(self):
        """Verify HostGalaxyTemplate can be imported from host."""
        from quasar_models.host import HostGalaxyTemplate
        assert HostGalaxyTemplate is not None

    def test_import_path_to_cache(self):
        """Verify PATH_TO_CACHE can be imported from host."""
        from quasar_models.host import PATH_TO_CACHE
        assert PATH_TO_CACHE is not None

    def test_import_path_to_data(self):
        """Verify PATH_TO_DATA can be imported from host."""
        from quasar_models.host import PATH_TO_DATA
        assert PATH_TO_DATA is not None


class TestIronModuleImports:
    """Test imports from quasar_models.iron submodule."""

    def test_import_iron_model(self):
        """Verify IronModel can be imported from iron."""
        from quasar_models.iron import IronModel
        assert IronModel is not None

    def test_import_iron_template(self):
        """Verify IronTemplate can be imported from iron."""
        from quasar_models.iron import IronTemplate
        assert IronTemplate is not None

    def test_import_path_to_cache(self):
        """Verify PATH_TO_CACHE can be imported from iron."""
        from quasar_models.iron import PATH_TO_CACHE
        assert PATH_TO_CACHE is not None

    def test_import_path_to_data(self):
        """Verify PATH_TO_DATA can be imported from iron."""
        from quasar_models.iron import PATH_TO_DATA
        assert PATH_TO_DATA is not None


class TestLineModuleImports:
    """Test imports from quasar_models.line submodule."""

    def test_import_gaussian_model(self):
        """Verify GaussianModel can be imported from line."""
        from quasar_models.line.line import GaussianModel
        assert GaussianModel is not None

    def test_import_vprofilecopy(self):
        """Verify VProfileCopy can be imported from line."""
        from quasar_models.line.line import VProfileCopy
        assert VProfileCopy is not None

    def test_import_vprofilecopy_1g(self):
        """Verify VProfileCopy1G can be imported from line."""
        from quasar_models.line.line import VProfileCopy1G
        assert VProfileCopy1G is not None

    def test_import_vprofilecopy_2g(self):
        """Verify VProfileCopy2G can be imported from line."""
        from quasar_models.line.line import VProfileCopy2G
        assert VProfileCopy2G is not None

    def test_import_vprofilecopy_3g(self):
        """Verify VProfileCopy3G can be imported from line."""
        from quasar_models.line.line import VProfileCopy3G
        assert VProfileCopy3G is not None

    def test_import_vprofilecopy_4g(self):
        """Verify VProfileCopy4G can be imported from line."""
        from quasar_models.line.line import VProfileCopy4G
        assert VProfileCopy4G is not None

    def test_import_vprofilecopy_5g(self):
        """Verify VProfileCopy5G can be imported from line."""
        from quasar_models.line.line import VProfileCopy5G
        assert VProfileCopy5G is not None


class TestLineVProfileCopiesModuleImports:
    """Test imports from quasar_models.line.vprofilecopies submodule."""

    def test_import_vprofilecopy_1g(self):
        """Verify VProfileCopy1G can be imported from vprofilecopies."""
        from quasar_models.line.vprofilecopies import VProfileCopy1G
        assert VProfileCopy1G is not None

    def test_import_vprofilecopy_2g(self):
        """Verify VProfileCopy2G can be imported from vprofilecopies."""
        from quasar_models.line.vprofilecopies import VProfileCopy2G
        assert VProfileCopy2G is not None

    def test_import_vprofilecopy_3g(self):
        """Verify VProfileCopy3G can be imported from vprofilecopies."""
        from quasar_models.line.vprofilecopies import VProfileCopy3G
        assert VProfileCopy3G is not None

    def test_import_vprofilecopy_4g(self):
        """Verify VProfileCopy4G can be imported from vprofilecopies."""
        from quasar_models.line.vprofilecopies import VProfileCopy4G
        assert VProfileCopy4G is not None

    def test_import_vprofilecopy_5g(self):
        """Verify VProfileCopy5G can be imported from vprofilecopies."""
        from quasar_models.line.vprofilecopies import VProfileCopy5G
        assert VProfileCopy5G is not None

    def test_import_vprofilecopy_base(self):
        """Verify _VProfileCopy can be imported from vprofilecopies."""
        from quasar_models.line.vprofilecopies import _VProfileCopy
        assert _VProfileCopy is not None


class TestModelingModuleImports:
    """Test imports from quasar_models.modeling submodule."""

    def test_import_base_model(self):
        """Verify BaseModel can be imported from modeling."""
        from quasar_models.modeling import BaseModel
        assert BaseModel is not None

    def test_import_fitter(self):
        """Verify Fitter can be imported from modeling."""
        from quasar_models.modeling import Fitter
        assert Fitter is not None

    def test_import_mcmc_fitter(self):
        """Verify MCMCFitter can be imported from modeling."""
        from quasar_models.modeling import MCMCFitter
        assert MCMCFitter is not None

    def test_import_linear_tie(self):
        """Verify LinearTie can be imported from modeling."""
        from quasar_models.modeling import LinearTie
        assert LinearTie is not None

    def test_import_prepare_model(self):
        """Verify PrepareModel can be imported from modeling."""
        from quasar_models.modeling import PrepareModel
        assert PrepareModel is not None

    def test_import_prepare_model_function(self):
        """Verify prepare_model function can be imported from modeling."""
        from quasar_models.modeling import prepare_model
        assert callable(prepare_model)

    def test_import_sequential_model(self):
        """Verify SequentialModel can be imported from modeling."""
        from quasar_models.modeling import SequentialModel
        assert SequentialModel is not None

    def test_import_prepared_sequential_model(self):
        """Verify PreparedSequentialModel can be imported from modeling."""
        from quasar_models.modeling import PreparedSequentialModel
        assert PreparedSequentialModel is not None

    def test_import_interpolated_sequential_model(self):
        """Verify InterpolatedSequentialModel can be imported from modeling."""
        from quasar_models.modeling import InterpolatedSequentialModel
        assert InterpolatedSequentialModel is not None


class TestModelingFittingModuleImports:
    """Test imports from quasar_models.modeling.fitting submodule."""

    def test_import_fitter(self):
        """Verify Fitter can be imported from modeling.fitting."""
        from quasar_models.modeling.fitting import Fitter
        assert Fitter is not None

    def test_import_mcmc_fitter(self):
        """Verify MCMCFitter can be imported from modeling.fitting."""
        from quasar_models.modeling.fitting import MCMCFitter
        assert MCMCFitter is not None


class TestModelingTemplateModuleImports:
    """Test imports from quasar_models.modeling.template submodule."""

    def test_import_base_template(self):
        """Verify BaseTemplate can be imported from modeling.template."""
        from quasar_models.modeling.template import BaseTemplate
        assert BaseTemplate is not None

    def test_import_template_model(self):
        """Verify TemplateModel can be imported from modeling.template."""
        from quasar_models.modeling.template import TemplateModel
        assert TemplateModel is not None
