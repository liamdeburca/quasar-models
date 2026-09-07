from Cython.Build import cythonize
from numpy import get_include
from setuptools import Extension, find_packages, setup

setup(
    name="quasar_models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        [
            # Array utils
            Extension(
                "quasar_models._core.utils",
                sources=["src/quasar_models/_core/utils.pyx"],
            ),
            # Powerlaw
            Extension(
                "quasar_models._core.modeling.powerlaw.evaluate",
                sources=["src/quasar_models/_core/modeling/powerlaw/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.powerlaw.fit_deriv",
                sources=["src/quasar_models/_core/modeling/powerlaw/fit_deriv.pyx"],
            ),
            # Gaussian
            Extension(
                "quasar_models._core.modeling.gaussian.evaluate",
                sources=["src/quasar_models/_core/modeling/gaussian/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.gaussian.fit_deriv",
                sources=["src/quasar_models/_core/modeling/gaussian/fit_deriv.pyx"],
            ),
            # VProfileCopy
            Extension(
                "quasar_models._core.modeling.vprofilecopy.evaluate",
                sources=["src/quasar_models/_core/modeling/vprofilecopy/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.vprofilecopy.fit_deriv",
                sources=["src/quasar_models/_core/modeling/vprofilecopy/fit_deriv.pyx"],
            ),
            # Convolution
            Extension(
                "quasar_models._core.convolution.utils",
                sources=["src/quasar_models/_core/convolution/utils.pyx"],
            ),
            # Template
            Extension(
                "quasar_models._core.modeling.template.evaluate",
                sources=["src/quasar_models/_core/modeling/template/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.template.fit_deriv",
                sources=["src/quasar_models/_core/modeling/template/fit_deriv.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.template.cytemplate",
                sources=["src/quasar_models/_core/modeling/template/cytemplate.pyx"],
            ),
            # Host
            Extension(
                "quasar_models._core.modeling.host.evaluate",
                sources=["src/quasar_models/_core/modeling/host/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.host.fit_deriv",
                sources=["src/quasar_models/_core/modeling/host/fit_deriv.pyx"],
            ),
            # Split
            Extension(
                "quasar_models._core.modeling.split.evaluate",
                sources=["src/quasar_models/_core/modeling/split/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.split.fit_deriv",
                sources=["src/quasar_models/_core/modeling/split/fit_deriv.pyx"],
            ),
            # Iron
            Extension(
                "quasar_models._core.modeling.iron.evaluate",
                sources=["src/quasar_models/_core/modeling/iron/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.iron.fit_deriv",
                sources=["src/quasar_models/_core/modeling/iron/fit_deriv.pyx"],
            ),
            # Balmer - Continuum
            Extension(
                "quasar_models._core.modeling.balmer.continuum.evaluate",
                sources=[
                    "src/quasar_models/_core/modeling/balmer/continuum/evaluate.pyx"
                ],
            ),
            # Balmer - Series
            Extension(
                "quasar_models._core.modeling.balmer.series.evaluate",
                sources=["src/quasar_models/_core/modeling/balmer/series/evaluate.pyx"],
            ),
            # Balmer
            Extension(
                "quasar_models._core.modeling.balmer.evaluate",
                sources=["src/quasar_models/_core/modeling/balmer/evaluate.pyx"],
            ),
            Extension(
                "quasar_models._core.modeling.balmer.fit_deriv",
                sources=["src/quasar_models/_core/modeling/balmer/fit_deriv.pyx"],
            ),
        ],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "warn.undeclared": False,
            "warn.unreachable": False,
        },
        force=True,  # Tabula rasa
        include_path=["src"],  # Necessary for cimporing .pxd files
    ),
    include_dirs=[get_include()],
    package_data={"quasar_models._core": ["**/*.pxd"]},
)