import numpy as np
from numpy import float64
from numpy.typing import NDArray
from astropy.units import Unit
import sqlite3
import pandas
from functools import lru_cache
from pathlib import Path

from quasar_utils.setup.units import UnitsInfo
from quasar_utils.setup import Info

from quasar_models.balmer.balmer_template import BalmerTemplate
from quasar_models.balmer.evaluation import evaluate

uinfo: UnitsInfo = UnitsInfo(
    wavelength_unit=Unit('1 angstrom'),
    velocity_unit=Unit('1 km/s'),
    temp_unit=Unit('1 K'),
    dens_unit=Unit('1 cm^-3'),
)
info: Info = Info(units_info=uinfo)

class QSFit:
    sigma_res: float = info.loading['sigma_res']
    
    edge: float = 3645
    temp: float = 15_000
    dens: float = 1e9
    tau: float = 1.0
    scale: float = 3.0

    rec_case: str = 'B'
    n_l: int = 7
    n_u: int = 50

    fwhm_bounds: tuple[float, float] = (1000.0, 20_000.0)
    fwhm_step: float = 250

    path_to_db: Path = \
        Path("/Users/liam/projects/storey_hummer_1995/databases/db.db")

    @classmethod
    @lru_cache(maxsize=1)
    def get_x_array(cls) -> NDArray[float64]:
        n = int(np.ceil(np.log(5) / np.log(1 + cls.sigma_res)))
        edges = 1.3 * cls.edge / (1 + cls.sigma_res)**np.arange(n)[::-1]
        return edges[:-1] * (1 + cls.sigma_res / 2)
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_fwhm_array(cls) -> NDArray[float64]:
        arr = np.arange(*cls.fwhm_bounds, cls.fwhm_step)
        if arr[-1] != cls.fwhm_bounds[1]:
            arr = np.append(arr, cls.fwhm_bounds[1])
        return info.units.getC(arr * Unit('km/s'))

    @classmethod
    @lru_cache(maxsize=1)
    def query_sh1995(cls) -> tuple[NDArray[float64], NDArray[float64]]:
        sql: str = \
            """
            SELECT wave as waves, val / SUM(val) OVER () as weights, n_u, temp, dens
            FROM emi
            WHERE z=1 AND n_l=2 AND n_u BETWEEN {} AND {} AND rec_case='{}' AND temp={} AND dens={}
            ORDER BY waves ASC
            """.format(cls.n_l, cls.n_u, cls.rec_case, cls.temp, cls.dens)
        
        with sqlite3.connect(cls.path_to_db) as conn:
            df = pandas.read_sql_query(sql, conn)

        waves = df['waves'].to_numpy(dtype=float64)
        weights = df['weights'].to_numpy(dtype=float64)

        return waves, weights

    @classmethod
    @lru_cache(maxsize=256)
    def forward(cls, fwhm: float) -> NDArray[float64]:
        x = cls.get_x_array()
        waves, weights = cls.query_sh1995()

        return evaluate(
            x,
            1.0, fwhm, cls.temp, cls.tau, cls.scale, cls.ratio,
            sigma_res=cls.sigma_res, edge=cls.edge, waves=waves, weights=weights,
            boltz=info.units.getBoltzmannFactor(), x_grid=x,
        )
        
    @classmethod
    def initialise(cls) -> None:
        assert hasattr(cls, 'ratio') and hasattr(cls, 'name')

        x_arr = cls.get_x_array()
        fwhm_arr = cls.get_fwhm_array()
        data = np.stack([cls.forward(fwhm) for fwhm in fwhm_arr], axis=0)

        waves, weights = cls.query_sh1995()

        cls.template: BalmerTemplate = BalmerTemplate(
            fwhm_arr, x_arr, data,
            edge=cls.edge,
            temp=cls.temp,
            dens=cls.dens,
            tau=cls.tau,
            scale=cls.scale,
            ratio=cls.ratio,
            waves=waves,
            weights=weights,
            case=cls.rec_case,
            n_l=cls.n_l,
            n_u=cls.n_u,
            info=info,
            is_logspace=True,
            based_on_template=True,
            name=cls.name,
        )
        cls.template.save(cls.name, overwrite=True)

class QSFit01(QSFit):
    ratio: float = 0.1
    name: str = 'qsfit_01'

class QSFit03(QSFit):
    ratio: float = 0.3
    name: str = 'qsfit_03'

class QSFit05(QSFit):
    ratio: float = 0.5
    name: str = 'qsfit_05'

class QSFit07(QSFit):
    ratio: float = 0.7
    name: str = 'qsfit_07'

class QSFit10(QSFit):
    ratio: float = 1.0
    name: str = 'qsfit_10'

def main_silent() -> None:
    QSFit01.initialise()
    QSFit03.initialise()
    QSFit05.initialise()
    QSFit07.initialise()
    QSFit10.initialise()

def main_verbose() -> None:
    for cls in [QSFit01, QSFit03, QSFit05, QSFit07, QSFit10]:
        print(f"Initialising: {cls.name}...", end='\r')
        try:
            cls.initialise()
            print(f"Initialising: {cls.name}... Success!")
        except:
            print(f"Initialising: {cls.name}... Failed!")

def main(silent: bool = False) -> None:
    main_silent() if silent else main_verbose()

def plot(name: str) -> None:
    template: BalmerTemplate = BalmerTemplate.load(name, info=info)

    import matplotlib.pyplot as plt
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    def transform(fwhm):
        return info.units.getC(fwhm).to('1e3km/s').value
    
    norm = Normalize(
        vmin=transform(template.fwhm[0]), 
        vmax=transform(template.fwhm[-1]),
    )
    scalmap = ScalarMappable(norm=norm, cmap=cmap)

    sel = slice(None, None, 10)

    fig, ax = plt.subplots(1, 1, sharex=True, dpi=300, figsize=(6, 4))
    fig.subplots_adjust(hspace=0)
    ax.set_title("Iron Emission Templates [upsampled]", loc='left')

    ax.text(
        0.95, 0.95, 
        template.name, 
        ha='right', va='top', transform=ax.transAxes,
    )
    for y, fwhm in zip(template.data[sel], transform(template.fwhm[sel])):
        ax.fill_between(
            template.x, y, 
            template.data[-1], 
            step='mid', color=scalmap.to_rgba(fwhm),
        )

    ax.set_xlabel(
        f"Rest wavelength ({info.units['wavelength_unit'].to_string()})",
        loc = 'right',
    )
    ax.set_ylabel('Flux density (a.u.)')

    ax.set_xlim(template.x[0], template.x[-1])
    ax.set_ylim(0)

    cbar = plt.colorbar(scalmap, ax=ax,)
    cbar.ax.yaxis.set_label_text(r"FWHM ($10^3$ km/s)")
    cbar.set_ticks([1, 5, 10, 15, 20])

    plt.show()

if __name__ == '__main__':
    main(silent=False)

    plot('qsfit_01')