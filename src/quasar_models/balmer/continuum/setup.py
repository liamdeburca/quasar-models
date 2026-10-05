"""
This script contains utilities for setting up various Balmer continuum templates.
"""

from itertools import product
from typing import ClassVar

from astropy.units import Quantity, Unit
from numpy import arange, log
from quasar_utils.setup import Info

from quasar_models.balmer.continuum import io
from quasar_models.balmer.continuum.balmer_continuum_template import (
    BalmerContinuumTemplate,
)

INFO = Info()


class QSFIT:
    X_BOUNDS: ClassVar[Quantity] = (1000.0, 4500.0) * Unit("angstrom")
    TEMPS: ClassVar[Quantity] = [
        10_000.0,
        12_500.0,
        15_000.0,
        20_000.0,
        30_000.0,
    ] * Unit("K")
    TAUS: ClassVar[tuple[float, ...]] = (1.0,)
    SCALES: ClassVar[tuple[float, ...]] = (3.0,)

    @classmethod
    def main(cls) -> None:
        cls.init_basic()

    @classmethod
    def get_x(cls):
        lb, ub = INFO.units.getUnitlessWavelength(cls.X_BOUNDS)
        n = int(log(ub / lb) / log(1 + INFO.loading.sigma_res) + 1)
        return lb * (1 + INFO.loading.sigma_res) ** arange(n + 1)

    @classmethod
    def get_fwhm(cls):
        return INFO.units.getUnitlessKMS(
            arange(1000, 20_000 + 1, 250) * Unit("km/s")
        )

    @classmethod
    def create_template(cls, temp: int, tau: float, scale: float) -> None:
        template = BalmerContinuumTemplate.instantiate(
            fwhm=cls.get_fwhm(),
            x=cls.get_x(),
            temp=temp,
            tau=tau,
            scale=scale,
            sigma_res=INFO.loading.sigma_res,
            n_scales=INFO.convolution.n_scales,
            edge=INFO.balmer.edge,
            fwhm_norm=INFO.balmer.fwhm_norm,
            boltz=INFO.units.getBoltzmannFactor(),
            is_logspace=True,
            name="qsfit",
        ).normalise()
        template.save_to_cache(INFO)

    @classmethod
    def init_basic(cls) -> None:
        for temp, tau, scale in product(
            INFO.units.getUnitlessTemperature(cls.TEMPS), 
            cls.TAUS, 
            cls.SCALES,
        ):
            cls.create_template(temp, tau, scale)

def main():
    QSFIT.main()


def plot() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=1.0, vmax=20.0)
    scalmap = ScalarMappable(norm=norm, cmap=cmap)
    sel = slice(None, None, 10)

    for path in io.PATH_TO_CACHE.glob("continuum*.fits"):
        template = BalmerContinuumTemplate.load(path=path, info=INFO)

        _, ax = plt.subplots(dpi=300, figsize=(8, 4))
        ax.set_title(path.stem, loc="left")

        for y, fwhm in zip(template.data[sel], template.fwhm[sel]):
            ax.fill_between(
                template.x,
                y,
                template.data[-1],
                step="mid",
                color=scalmap.to_rgba(fwhm / 1e3),
            )

        ax.set_xlabel(
            r"$\lambda_{\mathrm{rest}}$ ("
            + INFO.units.wavelength_unit.to_string()
            + ")",
            loc="right",
        )
        ax.set_ylabel("Flux density (a.u.)")
        ax.set_ylim(0)

        cbar = plt.colorbar(scalmap, ax=ax)
        cbar.ax.yaxis.set_label_text(r"FWHM ($10^3$ km/s)")
        cbar.set_ticks([1, 5, 10, 15, 20])

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
