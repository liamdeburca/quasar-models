from numpy import arange, empty, array, float64
from pathlib import Path
from astropy.io.fits import open as fits_open
from astropy.units import Unit

from quasar_models.iron.iron_template import IronTemplate, PATH_TO_CACHE
from quasar_models.iron.utils import _get_xlog

from quasar_utils.setup import Info

_this_file: Path = Path(__file__).resolve()
PATH_TO_DATA_DIR: Path = _this_file.parents[1] / 'src/quasar_models/iron/data'

class VestergaardWilkes2001:
    sigma_res: float = 1e-4 # ~30 km/s
    x_bounds: tuple[float, float] = (1000.0, 4000.0)

    path_to_data: Path = PATH_TO_DATA_DIR / "Fe_UVtmplt_A_im.fits"
    path: Path = PATH_TO_CACHE / "vw_2001.fits"

    template: IronTemplate | None = None

    @classmethod
    def initialise(cls) -> None:
        x_log = _get_xlog(cls.x_bounds, cls.sigma_res)
        info = Info()

        with fits_open(cls.path_to_data) as hdul:

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
            info.units['wavelength_unit'] = Unit('1 angstrom')
            info.units['velocity_unit'] = Unit('1 km/s')

            template: IronTemplate = IronTemplate(
                info.units.getC(fwhm * info.units['velocity_unit']),
                info.units.getWavelength(x * info.units['wavelength_unit']),
                data,
                info = info,
                is_logspace = False,
                name = "Vestergaard & Wilkes (2001)",
                path = cls.path,
            )
            cls.fwhm = template.fwhm
            # Transform and upsample
            _template = template.createLogspace(x_log)
            _template.resample(template.fwhm, inplace=True)
            template.mimicLogspace(_template, inplace=True)
            template.data /= template.data[-1].max()
            # Save to cache
            template.save(cls.path, overwrite=True)

            cls.template = template

class Veron2003:
    sigma_res: float = 1e-4 # ~30 km/s
    x_bounds: tuple[float, float] = (3000.0, 8000.0)
    path_to_data: Path = PATH_TO_DATA_DIR / "Fe2_Synth_Opt_tmplt_nrm.fits"
    path: Path = PATH_TO_CACHE / "v_2003.fits"

    template: IronTemplate | None = None

    @classmethod
    def initialise(cls) -> None:
        x_log = _get_xlog(cls.x_bounds, cls.sigma_res)
        info = Info()

        with fits_open(cls.path_to_data) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr['CRVAL1'] + hdr['CDELT1'] * arange(hdr['NAXIS1'])
            
            fwhm = empty(hdr['NAXIS2'], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith('APERT'): continue

                elems = [
                    elem \
                    for elem in line.strip().split(' ') \
                    if len(elem) > 0
                ]
                fwhm[int(elems[0]) - 1] = float(elems[1])

            # Adjust units to fit data
            info.units['wavelength_unit'] = Unit('1 angstrom')
            info.units['velocity_unit'] = Unit('1 km/s')

            template: IronTemplate = IronTemplate(
                info.units.getC(fwhm * info.units['velocity_unit']),
                info.units.getWavelength(x * info.units['wavelength_unit']),
                data,
                info = info,
                name = "Veron et al. (2003)",
                path = cls.path,
            )
            # Transform and upsample
            _template = template.createLogspace(x_log)
            _template.resample(template.fwhm, inplace=True)
            template.mimicLogspace(_template, inplace=True)
            template.data /= template.data[-1].max()
            # Save to cache
            template.save(cls.path, overwrite=True)

            cls.template = template

class BevWills:
    sigma_res: float = 1e-4 # ~30 km/s
    x_bounds: tuple[float, float] = (2800, 3800)
    path_to_data: Path = PATH_TO_DATA_DIR / "Fe_3100_izw1_BevWills.txt"
    path: Path = PATH_TO_CACHE / "bw.fits"

    template: IronTemplate | None = None

    @classmethod
    def initialise(cls) -> None:
        assert VestergaardWilkes2001.template is not None, \
            "Vestergaard & Wilkes (2001) template must be initialised first"
        
        x_log = _get_xlog(cls.x_bounds, cls.sigma_res)
        info = Info()

        fwhm = [900]

        x = []
        data = []
        with open(cls.path_to_data) as f:
            for x_str, data_str in map(str.split, f.readlines()):
                x.append(float(x_str))
                data.append(max(float(data_str), 0))

        x = array(x, dtype=float64)
        data = array(data)[None,:]
        data /= data[0].max()

        # Adjust units to fit data
        info.units['wavelength_unit'] = Unit('1 angstrom')
        info.units['velocity_unit'] = Unit('1 km/s')

        template: IronTemplate = IronTemplate(
            info.units.getC(fwhm * info.units['velocity_unit']),
            info.units.getWavelength(x * info.units['wavelength_unit']),
            data,
            info = info,
            name = 'bw',
            path = cls.path,
        )
        # Transform and upsample
        _template = template.createLogspace(x_log)
        _template.resample(VestergaardWilkes2001.template.fwhm, inplace=True)
        template.mimicLogspace(_template, inplace=True)
        template.data /= template.data[-1].max()
        # Save to cache
        template.save(cls.path, overwrite=True)

        cls.template = template

def main_silent() -> None:
    VestergaardWilkes2001.initialise()
    Veron2003.initialise()
    BevWills.initialise()

def main_verbose() -> None:
    print("Initialising: Vestergaard & Wilkes (2001)...", end='\r')
    try:
        VestergaardWilkes2001.initialise()
        print("Initialising: Vestergaard & Wilkes (2001)... Success!")
    except:
        print("Initialising: Vestergaard & Wilkes (2001)... Failed!")

    print("Initialising: Veron et al. (2003)...", end='\r')
    try:
        Veron2003.initialise()
        print("Initialising: Veron et al. (2003)... Success!")
    except:
        print("Initialising: Veron et al. (2003)... Failed!")

    print("Initialising: BevWills...", end='\r')
    try:
        BevWills.initialise()
        print("Initialising: BevWills... Success!")
    except:
        print("Initialising: BevWills... Failed!")

def main(silent: bool = False) -> None:
    if silent: main_silent()
    else:      main_verbose()

def plot() -> None:
    info = Info()
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    def transform(fwhm):
        return info.units.getC(fwhm).to('1e3km/s').value
    
    vw_2001 = VestergaardWilkes2001.template
    v_2003 = Veron2003.template
    bw = BevWills.template

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
    t = IronTemplate.load(vw_2001.path, info=info)
    ax.text(0.95, 0.95, t.name, ha='right', va='top', transform=ax.transAxes)

    for y, fwhm in zip(t.data[sel], transform(t.fwhm[sel])):
        ax.fill_between(t.x, y, t.data[-1], step='mid', color=scalmap.to_rgba(fwhm))

    ax = axes[1]
    t = IronTemplate.load(v_2003.path, info=info)
    ax.text(0.95, 0.95, t.name, ha='right', va='top', transform=ax.transAxes)

    for y, fwhm in zip(t.data[sel], transform(t.fwhm[sel])):
        ax.fill_between(t.x, y, t.data[-1], step='mid', color=scalmap.to_rgba(fwhm))

    ax = axes[2]
    t = IronTemplate.load(bw.path, info=info)
    ax.text(0.95, 0.95, t.name, ha='right', va='top', transform=ax.transAxes)

    for y, fwhm in zip(t.data[sel], transform(t.fwhm[sel])):
        ax.fill_between(t.x, y, t.data[-1], step='mid', color=scalmap.to_rgba(fwhm))

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
    main(silent=False)
    plot()