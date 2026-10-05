"""
This script contains utilities for setting up various host galaxy templates.

It currently supports data from the following:
- Bruzual & Charlot (2003), a.k.a. 'bc2003'.
"""

from pathlib import Path
from typing import ClassVar

from astropy.io import fits
from astropy.units import Quantity, Unit
from numpy import float64
from quasar_utils.setup import Info

from quasar_models.host import io
from quasar_models.host.host_galaxy_template import HostGalaxyTemplate

INFO: Info = Info()


class BC2003:
    NAME: ClassVar[str] = "bc2003"
    X_BOUNDS: ClassVar[Quantity] = (1000.0, 9500.0) * Unit("angstrom")
    PATH_TO_DATA: ClassVar[Path] = io.PATH_TO_DATA

    @classmethod
    def main(cls) -> None:
        cls.init_basic()

    @classmethod
    def get_paths(cls) -> list[Path]:
        return sorted(cls.PATH_TO_DATA.glob("tau06_z02_*_001.fits"))

    @classmethod
    def get_age_from_path(cls, path: Path) -> int:
        return int(
            path.name\
                .removeprefix("tau06_z02_")\
                .removesuffix("_001.fits")
        ) * 1000

    @classmethod
    def create_template(cls, path: Path) -> None:
        x_bounds = INFO.units.getUnitlessWavelength(cls.X_BOUNDS) 
        age = cls.get_age_from_path(path)

        with fits.open(path) as hdul:
            x = hdul[1].data["WAVELENGTH"].astype(float64)
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            fwhm = INFO.units.getUnitlessKMS([0] * Unit("km/s"))

            y = hdul[1].data["FLUX"].astype(float64)
            mask = (x_bounds[0] <= x) & (x <= x_bounds[1])

            x = x[mask]
            y = y[mask]

            template = HostGalaxyTemplate(
                fwhm=fwhm,
                x=x,
                data=y[None, :],
                is_logspace=False,
                name=cls.NAME,
                x_norm=INFO.host.x_norm,
                fwhm_norm=INFO.host.fwhm_norm,
                age=age,
            )
            template.normalise(inplace=True)
            template.save_to_cache(INFO)

    @classmethod
    def init_basic(cls) -> None:
        for path in cls.get_paths():
            cls.create_template(path)

def main() -> None:
    BC2003().main()


def plot() -> None:
    import matplotlib.pyplot as plt

    templates: list[HostGalaxyTemplate] = sorted(
        (
            HostGalaxyTemplate.load(path=path, info=INFO)
            for path in io.PATH_TO_CACHE.glob("bc2003*.fits")
        ),
        key=lambda t: t.age,
    )

    _, ax = plt.subplots(dpi=300, figsize=(8, 4))
    ax.set_title("Host Galaxy Templates", loc="left")

    for t in templates:
        ax.plot(t.x, t.data[0], label=f"{t.age / 1_000_000_000:.0f} Gyr", lw=1)

    ax.set_xlabel(
        r"$\lambda_{\mathrm{rest}}$ (" + INFO.units.wavelength_unit.to_string() + ")",
        loc="right",
    )
    ax.set_ylabel("Flux density (a.u.)")
    ax.set_ylim(0)
    ax.set_xlim(*INFO.units.getWavelength(BC2003.X_BOUNDS))
    ax.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the generated templates.",
    )
    args = parser.parse_args()

    main()
    if args.plot:
        plot()
