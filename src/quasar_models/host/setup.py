"""
This script contains utilities for setting up various host galaxy templates.

It currently supports data from the following:
- Bruzual & Charlot (2003), a.k.a. 'bc2003'.
"""

__all__ = ["BC2003"]

from pathlib import Path

from astropy.io import fits
from astropy.units import Unit
from numpy import array, float64
from quasar_typing.numpy import FloatVector
from quasar_utils.setup import Info

from quasar_models.host.host_galaxy_template import HostGalaxyTemplate
from quasar_models.host.io import PATH_TO_CACHE, PATH_TO_DATA


class BC2003:
    name: str = "bc2003"
    paths: list[Path] = sorted(PATH_TO_DATA.glob("tau06_z02_*_001.fits"))

    x_lb: float = 1000.0
    x_ub: float = 9500.0

    def __init__(self):
        self.info: Info = Info()
        self.fwhm: FloatVector = array([0], dtype=float64)
        self.__post_init__()

    def __post_init__(self):
        PATH_TO_CACHE.mkdir(exist_ok=True)

    @classmethod
    def get_age_from_path(cls, path: Path) -> int:
        return 1000 * int(
            path.name.removeprefix("tau06_z02_").removesuffix("_001.fits")
        )

    def main(self) -> None:
        for path in self.paths:
            age = self.get_age_from_path(path)

            with fits.open(path) as hdul:
                x = hdul[1].data["WAVELENGTH"].astype(float64)
                y = hdul[1].data["FLUX"].astype(float64)
                mask = (self.x_lb <= x) & (x <= self.x_ub)

                x = x[mask]
                y = y[mask]

                template = HostGalaxyTemplate(
                    fwhm=self.fwhm,
                    x=self.info.units.getWavelength(x * Unit("angstrom")),
                    data=y[None, :],
                    is_logspace=False,
                    name=self.name,
                    x_norm=self.info.host.x_norm,
                    fwhm_norm=self.info.host.fwhm_norm,
                    age=age,
                )
                template.normalise(inplace=True)
                template.save_to_cache(self.info)
                del template


def main() -> None:
    BC2003().main()


def plot() -> None:
    import matplotlib.pyplot as plt

    bc2003 = BC2003()
    info = bc2003.info

    templates: list[HostGalaxyTemplate] = sorted(
        (
            HostGalaxyTemplate.load(path=path, info=info)
            for path in PATH_TO_CACHE.glob("bc2003*.fits")
        ),
        key=lambda t: t.age,
    )

    fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
    ax.set_title("Host Galaxy Templates", loc="left")

    for t in templates:
        ax.plot(t.x, t.data[0], label=f"{t.age / 1_000_000_000:.0f} Gyr", lw=1)

    ax.set_xlabel(
        r"$\lambda_{\mathrm{rest}}$ (" + info.units.wavelength_unit.to_string() + ")",
        loc="right",
    )
    ax.set_ylabel("Flux density (a.u.)")
    ax.set_ylim(0)
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
