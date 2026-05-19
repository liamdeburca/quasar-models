from numpy import arange, empty, array, float64
from pathlib import Path
from astropy.io import fits
from astropy.units import Unit

from quasar_models.iron.iron_template import IronTemplate
from quasar_models.iron.utils import _get_xlog
from quasar_models.iron.io import PATH_TO_DATA

from quasar_utils.setup import Info

class VestergaardWilkes2001:
    def __init__(self):
        self.info: Info = Info()
        self.x_bounds: tuple[float, float] = (1000.0, 4000.0)
        self.path_to_data: Path = PATH_TO_DATA / "Fe_UVtmplt_A_im.fits"

        self.template: IronTemplate | None = None

    def main(self) -> None:
        x_log = _get_xlog(self.x_bounds, self.info.loading.sigma_res)

        with fits.open(self.path_to_data) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr['CRVAL1'] + hdr['CDELT1'] * arange(hdr['NAXIS1'])
            
            fwhm = empty(hdr['NAXIS2'], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith('APERT'): 
                    continue

                elems = [
                    elem \
                    for elem in line.strip().split(' ') \
                    if len(elem) > 0
                ]
                fwhm[int(elems[0]) - 1] = float(elems[1])

            # Adjust units to fit data
            self.info.units['wavelength_unit'] = Unit('1 angstrom')
            self.info.units['velocity_unit'] = Unit('1 km/s')

            self.template = IronTemplate(
                self.info.units.getC(fwhm * self.info.units['velocity_unit']),
                self.info.units.getWavelength(x * self.info.units['wavelength_unit']),
                data,
                info=self.info,
                is_logspace=False,
                name="vw2001",
            )

            _template = self.template.createLogspace(x_log)
            _template.resample(self.template.fwhm, inplace=True)
            self.template.mimicLogspace(_template, inplace=True)

            self.template.save_to_cache()

class Veron2003:
    def __init__(self):
        self.info: Info = Info()
        self.x_bounds: tuple[float, float] = (3000.0, 8000.0)
        self.path_to_data: Path = PATH_TO_DATA / "Fe2_Synth_Opt_tmplt_nrm.fits"

        self.template: IronTemplate | None = None

    def main(self) -> None:
        x_log = _get_xlog(self.x_bounds, self.info.loading.sigma_res)

        with fits.open(self.path_to_data) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr['CRVAL1'] + hdr['CDELT1'] * arange(hdr['NAXIS1'])
            
            fwhm = empty(hdr['NAXIS2'], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith('APERT'): 
                    continue

                elems = [
                    elem \
                    for elem in line.strip().split(' ') \
                    if len(elem) > 0
                ]
                fwhm[int(elems[0]) - 1] = float(elems[1])

            # Adjust units to fit data
            self.info.units['wavelength_unit'] = Unit('1 angstrom')
            self.info.units['velocity_unit'] = Unit('1 km/s')

            self.template = IronTemplate(
                self.info.units.getC(fwhm * self.info.units['velocity_unit']),
                self.info.units.getWavelength(x * self.info.units['wavelength_unit']),
                data,
                info=self.info,
                name="v2003",
                is_logspace=False,
            )

            _template = self.template.createLogspace(x_log)
            _template.resample(self.template.fwhm, inplace=True)
            self.template.mimicLogspace(_template, inplace=True)

            self.template.save_to_cache()

class BevWills:
    def __init__(self):
        self.info: Info = Info()
        self.x_bounds: tuple[float, float] = (2800, 3800)
        self.path_to_data: Path = PATH_TO_DATA / "Fe_3100_izw1_BevWills.txt"

        self.template: IronTemplate | None = None

    def main(self) -> None:        
        x_log = _get_xlog(self.x_bounds, self.info.loading.sigma_res)

        fwhm = [900]

        x = []
        data = []
        with open(self.path_to_data) as f:
            for x_str, data_str in map(str.split, f.readlines()):
                x.append(float(x_str))
                data.append(max(float(data_str), 0))

        x = array(x, dtype=float64)
        data = array(data)[None,:]
        data /= data[0].max()

        # Adjust units to fit data
        self.info.units['wavelength_unit'] = Unit('1 angstrom')
        self.info.units['velocity_unit'] = Unit('1 km/s')

        self.template = IronTemplate(
            self.info.units.getC(fwhm * self.info.units['velocity_unit']),
            self.info.units.getWavelength(x * self.info.units['wavelength_unit']),
            data,
            info = self.info,
            name = 'bw',
            is_logspace=False,
        )

        vw2001 = IronTemplate.load_from_cache("vw2001", info=self.info)

        _template = self.template.createLogspace(x_log)
        _template.resample(vw2001.fwhm, inplace=True)
        self.template.mimicLogspace(_template, inplace=True)
        self.template.data /= self.template.data[-1].max()

        self.template.save_to_cache()

def main_silent() -> None:
    VestergaardWilkes2001().main()
    Veron2003().main()
    BevWills().main()

def main_verbose() -> None:
    print("Initialising: Vestergaard & Wilkes (2001)...", end='\r')
    try:
        VestergaardWilkes2001().main()
        print("Initialising: Vestergaard & Wilkes (2001)... Success!")
    except Exception:
        print("Initialising: Vestergaard & Wilkes (2001)... Failed!")

    print("Initialising: Veron et al. (2003)...", end='\r')
    try:
        Veron2003().main()
        print("Initialising: Veron et al. (2003)... Success!")
    except Exception:
        print("Initialising: Veron et al. (2003)... Failed!")

    print("Initialising: BevWills...", end='\r')
    try:
        BevWills().main()
        print("Initialising: BevWills... Success!")
    except Exception:
        print("Initialising: BevWills... Failed!")

def main(silent: bool = False) -> None:
    (main_silent if silent else main_verbose)()

def plot() -> None:
    info = Info()
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    def transform(fwhm):
        return info.units.getC(fwhm).to('1e3km/s').value
    
    vw_2001 = IronTemplate.load_from_cache("vw2001", info=info)
    v_2003 = IronTemplate.load_from_cache("v2003", info=info)
    bw = IronTemplate.load_from_cache("bw", info=info)

    norm = Normalize(
        vmin=transform(vw_2001.fwhm[0]), 
        vmax=transform(vw_2001.fwhm[-1]),
    )
    scalmap = ScalarMappable(norm=norm, cmap=cmap)

    sel = slice(None, None, 10)

    fig, axes = plt.subplots(3, 1, sharex=True, dpi=300, figsize=(8, 4))
    fig.subplots_adjust(hspace=0)
    axes[0].set_title("Iron Emission Templates [upsampled]", loc='left')

    ax = axes[0]
    t = vw_2001
    ax.text(0.95, 0.95, t.name, ha='right', va='top', transform=ax.transAxes)

    for y, fwhm in zip(t.data[sel], transform(t.fwhm[sel])):
        ax.fill_between(t.x, y / t.normalisation, t.data[-1] / t.normalisation, step='mid', color=scalmap.to_rgba(fwhm))

    ax = axes[1]
    t = v_2003
    ax.text(0.95, 0.95, t.name, ha='right', va='top', transform=ax.transAxes)

    for y, fwhm in zip(t.data[sel], transform(t.fwhm[sel])):
        ax.fill_between(t.x, y / t.normalisation, t.data[-1] / t.normalisation, step='mid', color=scalmap.to_rgba(fwhm))

    ax = axes[2]
    t = bw
    ax.text(0.95, 0.95, t.name, ha='right', va='top', transform=ax.transAxes)

    for y, fwhm in zip(t.data[sel], transform(t.fwhm[sel])):
        ax.fill_between(t.x, y / t.normalisation, t.data[-1] / t.normalisation, step='mid', color=scalmap.to_rgba(fwhm))

    axes[2].set_xlabel(
        f"Rest wavelength ({info.units['wavelength_unit'].to_string()})",
        loc = 'right',
    )
    axes[1].set_ylabel('Flux density (a.u.)')

    for ax in axes:
        ax.set_ylim(0)
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0, 1.0, 2.0])

    cbar = plt.colorbar(
        scalmap,
        ax=axes,
    )
    cbar.ax.yaxis.set_label_text(r"FWHM ($10^3$ km/s)")
    cbar.set_ticks(
        [1, 5, 10, 15, 20]
    )

    plt.show()

if __name__ == "__main__":
    main(silent=True)
    plot()