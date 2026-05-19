"""
This script contains utilities for setting up various Balmer series templates.

It currently supports data from the following:
- Storey & Hummer (1995), a.k.a. 'sh1995'. 
"""
__all__ = ['SH1995']

from pathlib import Path
from pandas import DataFrame, read_csv
from numpy import arange, log, unique, isin
from astropy.units import Unit
from astropy.constants import c
from itertools import product

from quasar_utils.setup import Info
from quasar_typing.numpy import FloatVector

from quasar_models.balmer.series.io import PATH_TO_DATA, PATH_TO_CACHE
from quasar_models.balmer.series.balmer_series_template import BalmerSeriesTemplate

class SH1995:
    name: str = 'sh1995'
    path_to_csv: Path = PATH_TO_DATA / 'sh1995.csv'

    temp_range: list[float] = [
        10_000.0,
        12_500.0,
        15_000.0,
        20_000.0,
        30_000.0,
    ]
    dens_range: list[float] = [
        1e9,
    ]

    def __init__(self):
        self.data: DataFrame = read_csv(self.path_to_csv)
        self.n_u_range: tuple[int, int] = (
            self.data['n_u'].min(), 
            self.data['n_u'].max(),
        )
        self.info: Info = Info()

        self.fwhm: FloatVector = arange(1000, 20_000+1, 250) / c.to("km/s").value

        x0, x1 = self.info.units.getWavelength([1000, 4500] * Unit('angstrom'))
        n = int(log(x1 / x0) / log(1 + self.info.loading.sigma_res)) + 1
        self.x: FloatVector = x0 * (1 + self.info.loading.sigma_res)**arange(n+1)

        self._temp: FloatVector = unique(self.data['temp'])
        self._dens: FloatVector = unique(self.data['dens'])

        self.__post_init__()

    def __post_init__(self):
        PATH_TO_CACHE.mkdir(exist_ok=True)

        assert isin(self.temp_range, self._temp).all()
        assert isin(self.dens_range, self._dens).all()

    def get_waves_and_weights(
        self,
        *, 
        temp: float, dens: float, n_u_range: tuple[int, int],
    ) -> tuple[FloatVector, FloatVector]:
        mask = (
            (self.data['temp'] == temp) &
            (self.data['dens'] == dens) &
            (self.data['n_u'] >= n_u_range[0]) &
            (self.data['n_u'] <= n_u_range[1])
        )
        waves = self.info.units.getWavelength(
            self.data.loc[mask,'wave'].values * Unit('angstrom')
        )
        weights = self.data.loc[mask,'val'].values
        weights /= weights.sum()

        return waves, weights

    def get_template(
        self,
        *,
        temp: float, dens: float, n_u_range: tuple[int, int],
    ) -> BalmerSeriesTemplate:
        waves, weights = self.get_waves_and_weights(
            temp=temp, 
            dens=dens, 
            n_u_range=n_u_range,
        )
        template = BalmerSeriesTemplate.instantiate(
            self.fwhm, self.x,
            waves, weights, 
            temp, dens, n_u_range,
            info=self.info,
            is_logspace=True,
            name=self.name,
            normalisation=None,
        )
        return template.normalise(inplace=True)
    
    def main(self) -> None:
        for temp, dens in product(self.temp_range, self.dens_range):
            _ = self.get_template(
                temp=temp, 
                dens=dens, 
                n_u_range=self.n_u_range,
            ).save_to_cache()

def main() -> None:
    SH1995().main()

def plot() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable


    sh1995 = SH1995()
    info = sh1995.info
    
    def transform(fwhm):
        return info.units.getC(fwhm).to('km/s').value / 1e3
    
    norm = Normalize(
        vmin=transform(sh1995.fwhm[0]), 
        vmax=transform(sh1995.fwhm[-1])
    )
    scalmap = ScalarMappable(norm=norm, cmap=cmap)
    sel = slice(None, None, 10)

    for path in PATH_TO_CACHE.glob("series*.fits"):
        template = BalmerSeriesTemplate.load(path, info)

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

if __name__ == '__main__':
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