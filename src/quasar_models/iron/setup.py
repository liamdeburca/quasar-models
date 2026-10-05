from pathlib import Path
from typing import ClassVar

from astropy.io import fits
from astropy.units import Quantity, Unit
from numpy import arange, array, empty, float64
from quasar_utils.setup import Info

from quasar_models.iron import io
from quasar_models.iron.iron_template import IronTemplate
from quasar_models.iron.utils import _get_xlog

INFO: Info = Info()
PATH_TO_DATA: Path = io.PATH_TO_DATA / "Fe_UVtmplt_A_im.fits"

class VW2001:
    X_BOUNDS: ClassVar[Quantity] = (1000.0, 3250.0) * Unit("angstrom")
    RED_BLUE_DIVIDE: ClassVar[Quantity] = 2150.0 * Unit("angstrom")

    @classmethod
    def main(cls) -> None:
        cls.init_basic()
        cls.init_blue()
        cls.init_red()

    @classmethod
    def init_basic(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        with fits.open(PATH_TO_DATA) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr["CRVAL1"] + hdr["CDELT1"] * arange(hdr["NAXIS1"])
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            fwhm = empty(hdr["NAXIS2"], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith("APERT"):
                    continue

                elems = [elem for elem in line.strip().split(" ") if len(elem) > 0]
                fwhm[int(elems[0]) - 1] = float(elems[1])
            fwhm = INFO.units.getUnitlessKMS(fwhm * Unit("km/s"))

            template = IronTemplate(
                fwhm=fwhm, # Array already in km/s
                x=x, 
                data=data,
                is_logspace=False,
                name="vw2001",
                fwhm_norm=INFO.iron.fwhm_norm,
            )
            _template = template.createLogspace(
                sigma_res=INFO.loading.sigma_res,
                xr=x_log,
            )
            _template = _template.resample(template.fwhm)

            template.mimicLogspace(_template, inplace=True)
            template.simplifyData(tol=1e-8, inplace=True)

            template.save_to_cache(INFO)

    @classmethod
    def init_blue(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))
        red_blue_divide = INFO.units.getUnitlessWavelength(cls.RED_BLUE_DIVIDE)

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        with fits.open(PATH_TO_DATA) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr["CRVAL1"] + hdr["CDELT1"] * arange(hdr["NAXIS1"])
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            # Modify data
            data[:,red_blue_divide <= x] = 0.0

            fwhm = empty(hdr["NAXIS2"], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith("APERT"):
                    continue

                elems = [elem for elem in line.strip().split(" ") if len(elem) > 0]
                fwhm[int(elems[0]) - 1] = float(elems[1])
            fwhm = INFO.units.getUnitlessKMS(fwhm * Unit("km/s"))

            template = IronTemplate(
                fwhm=fwhm, # Array already in km/s
                x=x, 
                data=data,
                is_logspace=False,
                name="vw2001_blue",
                fwhm_norm=INFO.iron.fwhm_norm,
            )
            _template = template.createLogspace(
                sigma_res=INFO.loading.sigma_res,
                xr=x_log,
            )
            _template = _template.resample(template.fwhm)

            template.mimicLogspace(_template, inplace=True)
            template.simplifyData(tol=1e-8, inplace=True)

            template.save_to_cache(INFO)


    @classmethod
    def init_red(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))
        red_blue_divide = INFO.units.getUnitlessWavelength(cls.RED_BLUE_DIVIDE)

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        with fits.open(PATH_TO_DATA) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr["CRVAL1"] + hdr["CDELT1"] * arange(hdr["NAXIS1"])
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            # Modify data
            data[:,x < red_blue_divide] = 0.0

            fwhm = empty(hdr["NAXIS2"], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith("APERT"):
                    continue

                elems = [elem for elem in line.strip().split(" ") if len(elem) > 0]
                fwhm[int(elems[0]) - 1] = float(elems[1])
            fwhm = INFO.units.getUnitlessKMS(fwhm * Unit("km/s"))

            template = IronTemplate(
                fwhm=fwhm, # Array already in km/s
                x=x, 
                data=data,
                is_logspace=False,
                name="vw2001_red",
                fwhm_norm=INFO.iron.fwhm_norm,
            )
            _template = template.createLogspace(
                sigma_res=INFO.loading.sigma_res,
                xr=x_log,
            )
            _template = _template.resample(template.fwhm)

            template.mimicLogspace(_template, inplace=True)
            template.simplifyData(tol=1e-8, inplace=True)

            template.save_to_cache(INFO)


class V2003:
    X_BOUNDS: ClassVar[Quantity] = (3000.0, 8000.0) * Unit("angstrom")
    PATH_TO_DATA: ClassVar[Path] = io.PATH_TO_DATA / "Fe2_Synth_Opt_tmplt_nrm.fits"
    RED_BLUE_DIVIDE: ClassVar[Quantity] = 4800.0 * Unit("angstrom")

    @classmethod
    def main(cls) -> None:
        cls.init_basic()
        cls.init_blue()
        cls.init_red()

    @classmethod
    def init_basic(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        with fits.open(cls.PATH_TO_DATA) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr["CRVAL1"] + hdr["CDELT1"] * arange(hdr["NAXIS1"])
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            fwhm = empty(hdr["NAXIS2"], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith("APERT"):
                    continue

                elems = [elem for elem in line.strip().split(" ") if len(elem) > 0]
                fwhm[int(elems[0]) - 1] = float(elems[1])

            fwhm = INFO.units.getUnitlessKMS(fwhm * Unit("km/s"))

            template = IronTemplate(
                fwhm=fwhm,
                x=x,
                data=data,
                is_logspace=False,
                name="v2003",
                fwhm_norm=INFO.iron.fwhm_norm,
            )
            _template = template.createLogspace(
                sigma_res=INFO.loading.sigma_res,
                xr=x_log,
            )
            _template = _template.resample(template.fwhm)

            template.mimicLogspace(_template, inplace=True)
            template.simplifyData(tol=1e-8, inplace=True)

            template.save_to_cache(INFO)

    @classmethod
    def init_blue(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))
        red_blue_divide = INFO.units.getUnitlessWavelength(cls.RED_BLUE_DIVIDE)

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        with fits.open(cls.PATH_TO_DATA) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr["CRVAL1"] + hdr["CDELT1"] * arange(hdr["NAXIS1"])
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            # Modify data
            data[:,red_blue_divide <= x] = 0.0

            fwhm = empty(hdr["NAXIS2"], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith("APERT"):
                    continue

                elems = [elem for elem in line.strip().split(" ") if len(elem) > 0]
                fwhm[int(elems[0]) - 1] = float(elems[1])

            fwhm = INFO.units.getUnitlessKMS(fwhm * Unit("km/s"))

            template = IronTemplate(
                fwhm=fwhm,
                x=x,
                data=data,
                is_logspace=False,
                name="v2003_blue",
                fwhm_norm=INFO.iron.fwhm_norm,
            )
            _template = template.createLogspace(
                sigma_res=INFO.loading.sigma_res,
                xr=x_log,
            )
            _template = _template.resample(template.fwhm)

            template.mimicLogspace(_template, inplace=True)
            template.simplifyData(tol=1e-8, inplace=True)

            template.save_to_cache(INFO)

    @classmethod
    def init_red(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))
        red_blue_divide = INFO.units.getUnitlessWavelength(cls.RED_BLUE_DIVIDE)

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        with fits.open(cls.PATH_TO_DATA) as hdul:
            hdu = hdul[0]
            hdr = hdu.header

            data = hdu.data.astype(float64)
            data /= data[0].max()

            x = hdr["CRVAL1"] + hdr["CDELT1"] * arange(hdr["NAXIS1"])
            x = INFO.units.getUnitlessWavelength(x * Unit("angstrom"))

            # Modify data
            data[:,x < red_blue_divide] = 0.0

            fwhm = empty(hdr["NAXIS2"], dtype=float64)
            for key, line in hdr.items():
                if not key.startswith("APERT"):
                    continue

                elems = [elem for elem in line.strip().split(" ") if len(elem) > 0]
                fwhm[int(elems[0]) - 1] = float(elems[1])

            fwhm = INFO.units.getUnitlessKMS(fwhm * Unit("km/s"))

            template = IronTemplate(
                fwhm=fwhm,
                x=x,
                data=data,
                is_logspace=False,
                name="v2003_red",
                fwhm_norm=INFO.iron.fwhm_norm,
            )
            _template = template.createLogspace(
                sigma_res=INFO.loading.sigma_res,
                xr=x_log,
            )
            _template = _template.resample(template.fwhm)

            template.mimicLogspace(_template, inplace=True)
            template.simplifyData(tol=1e-8, inplace=True)

            template.save_to_cache(INFO)

class BW:
    X_BOUNDS: ClassVar[Quantity] = (2800, 3800) * Unit("angstrom")
    PATH_TO_DATA: ClassVar[Path] = io.PATH_TO_DATA / "Fe_3100_izw1_BevWills.txt"

    @classmethod
    def main(cls) -> None:
        cls.init_basic()

    @classmethod
    def init_basic(cls) -> None:
        x_bounds = tuple(INFO.units.getUnitlessWavelength(cls.X_BOUNDS))

        x_log = _get_xlog(x_bounds, INFO.loading.sigma_res)
        fwhm = INFO.units.getUnitlessKMS([900] * Unit("km/s"))

        x = []
        data = []
        with open(cls.PATH_TO_DATA) as f:
            for x_str, data_str in map(str.split, f.readlines()):
                x.append(float(x_str))
                data.append(max(float(data_str), 0))

        x = array(x, dtype=float64)
        x = INFO.units.getWavelength(x * Unit("angstrom"))

        data = array(data)[None, :]
        data /= data[0].max()

        template = IronTemplate(
            fwhm=fwhm,
            x=x,
            data=data,
            is_logspace=False,
            name="bw",
            fwhm_norm=INFO.iron.fwhm_norm,
        )
        _template = template.createLogspace(
            sigma_res=INFO.loading.sigma_res,
            xr=x_log,
        )
        vw2001 = IronTemplate.load_from_cache(name="vw2001", info=INFO)
        _template = _template.resample(vw2001.fwhm)

        template.mimicLogspace(_template, inplace=True)
        template.simplifyData(tol=1e-8, inplace=True)

        template.save_to_cache(INFO)


def main() -> None:
    VW2001().main()
    V2003().main()
    BW().main()


def plot() -> None:
    info = Info()

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize

    temps = [
        IronTemplate.load_from_cache(name=name, info=info)
        for name in ["vw2001", "vw2001_blue", "vw2001_red", "bw", "v2003", "v2003_blue", "v2003_red"]
    ]

    norm = Normalize(
        vmin=temps[0].fwhm[0] / 1e3,
        vmax=temps[0].fwhm[-1] / 1e3,
    )
    scalmap = ScalarMappable(norm=norm, cmap=cmap)

    sel = slice(None, None, 10)

    fig, axes = plt.subplots(len(temps), 1, sharex=True, sharey=True, dpi=300, figsize=(8, len(temps)))
    fig.subplots_adjust(hspace=0)
    axes[0].set_title("Iron Emission Templates [upsampled]", loc="left")

    for ax, t in zip(axes, temps):
        ax.text(0.95, 0.95, t.name, ha="right", va="top", transform=ax.transAxes)

        for y, fwhm in zip(t.data[sel], t.fwhm[sel]):
            ax.fill_between(
                t.x,
                y / t.normalisation,
                t.data[-1] / t.normalisation,
                step="mid",
                color=scalmap.to_rgba(fwhm / 1e3),
            )

    ax.set_xlabel(
        f"Rest wavelength ({info.units['wavelength_unit'].to_string()})",
        loc="right",
    )
    axes[len(axes) // 2].set_ylabel("Flux density (a.u.)")

    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 1.0, 2.0])

    cbar = plt.colorbar(
        scalmap,
        ax=axes,
    )
    cbar.ax.yaxis.set_label_text(r"FWHM ($10^3$ km/s)")
    cbar.set_ticks([1, 5, 10, 15, 20])

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialise iron templates.")
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    args = parser.parse_args()

    main()
    if args.plot:
        plot()
