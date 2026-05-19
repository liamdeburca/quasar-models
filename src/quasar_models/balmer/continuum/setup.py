"""
This script contains utilities for setting up various Balmer continuum templates.
"""
from pathlib import Path
from numpy import arange, exp, log
from astropy.units import Unit
from astropy.constants import c
from itertools import product

from quasar_utils.setup import Info
from quasar_typing.numpy import FloatVector

from quasar_models.balmer.continuum.io import PATH_TO_CACHE
from quasar_models.balmer.continuum.balmer_continuum_template import BalmerContinuumTemplate

class QSFit:
    name: str = 'qsfit'

    temp_range: list[float] = [
        10_000.0,
        12_500.0,
        15_000.0,
        20_000.0,
        30_000.0,
    ]
    tau_range: list[float] = [
        1.0,
    ]
    scale_range: list[float] = [
        3.0,
    ]

    def __init__(self):
        self.info: Info = Info()
        
        self.fwhm: FloatVector = arange(1000, 20_000+1, 250) / c.to("km/s").value

        x0, x1 = self.info.units.getWavelength([1000, 4500] * Unit('angstrom'))
        n = int(log(x1 / x0) / log(1 + self.info.loading.sigma_res)) + 1
        self.x: FloatVector = x0 * (1 + self.info.loading.sigma_res)**arange(n+1)

        self.__post_init__()

    def __post_init__(self):
        PATH_TO_CACHE.mkdir(exist_ok=True)

    def get_template(
        self,
        *,
        temp: float,
        tau: float,
        scale: float,
    ) -> BalmerContinuumTemplate:
        return BalmerContinuumTemplate.instantiate(
            self.fwhm, self.x,
            temp, tau, scale,
            info=self.info,
            is_logspace=True,
            name=self.name,
        )

    def main(self) -> None:
        for temp, tau, scale in product(self.temp_range, self.tau_range, self.scale_range):
            _ = self.get_template(
                temp=temp, 
                tau=tau, 
                scale=scale,
            ).save_to_cache()

def main():
    QSFit().main()

def plot() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable


    sh1995 = QSFit()
    info = sh1995.info
    
    def transform(fwhm):
        return info.units.getC(fwhm).to('km/s').value / 1e3
    
    norm = Normalize(
        vmin=transform(sh1995.fwhm[0]), 
        vmax=transform(sh1995.fwhm[-1])
    )
    scalmap = ScalarMappable(norm=norm, cmap=cmap)
    sel = slice(None, None, 10)

    for path in PATH_TO_CACHE.glob("continuum*.fits"):
        template = BalmerContinuumTemplate.load(path, info)

        fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
        ax.set_title(path.stem, loc='left')

        for y, fwhm in zip(template.data[sel], template.fwhm[sel]):
            ax.fill_between(
                template.x, 
                y, template.data[-1], 
                step='mid', 
                color=scalmap.to_rgba(transform(fwhm)),
            )

        ax.set_xlabel(
            r"$\lambda_{\mathrm{rest}}$ (" + info.units.wavelength_unit.to_string() + ")",
            loc='right',
        )
        ax.set_ylabel('Flux density (a.u.)')
        ax.set_ylim(0)

        cbar = plt.colorbar(scalmap, ax=ax)
        cbar.ax.yaxis.set_label_text(r"FWHM ($10^3$ km/s)")
        cbar.set_ticks([1, 5, 10, 15, 20])

        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot', action='store_true',
        help="Whether to plot the generated templates.",
    )
    args = parser.parse_args()

    main()
    if args.plot:
        plot()