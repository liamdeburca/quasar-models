from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.units import Unit
from quasar_utils.setup import Info

from quasar_models.iron.iron_template import IronTemplate
from quasar_models.iron.utils import _get_xlog


def plot_template(
    base_template: IronTemplate, template: IronTemplate, info: Info
) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.cm import rainbow as cmap
    from matplotlib.colors import Normalize

    def transform(fwhm):
        return info.units.getC(fwhm).to("1e3 km/s").value

    norm = Normalize(
        vmin=transform(template.fwhm[0]),
        vmax=transform(template.fwhm[-1]),
    )
    scalmap = ScalarMappable(norm=norm, cmap=cmap)

    sel = slice(None, None, 5)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), dpi=300, sharex=True, sharey=True)
    axes[0].set_title(
        f"Iron template: {template.name} from {base_template.name}", loc="left"
    )

    for ax, temp in zip(axes, (base_template, template)):
        for y, fwhm in zip(temp.data[sel], transform(temp.fwhm[sel])):
            ax.fill_between(
                temp.x,
                y / temp.normalisation,
                temp.data[-1] / temp.normalisation,
                step="mid",
                color=scalmap.to_rgba(fwhm),
            )
        ax.set_ylabel("Flux density (a.u.)")

    axes[-1].set_xlabel(
        f"Rest wavelength ({info.units.wavelength_unit.to_string()})",
        loc="right",
    )

    axes[0].set_ylim(0)
    cbar = plt.colorbar(scalmap, ax=axes)
    cbar.ax.yaxis.set_label_text(r"FWHM ($10^3$ km/s)")
    cbar.set_ticks([1, 5, 10, 15, 20])

    plt.show()


def main(args: Namespace) -> None:
    # COnfigure Info instance
    info = Info()
    info.units.wavelength_unit = Unit("1 Angstrom")
    info.units.velocity_unit = Unit("1 km/s")
    info.loading._sigma_res = args.v_res * Unit("km/s")
    info.force_update()

    assert info.loading.sigma_res == args.v_res / c.to("km/s").value

    base_template = IronTemplate.load_from_cache(
        args.base,
        info=info,
    )
    lb = args.lb or base_template.x[0]
    ub = args.ub or base_template.x[-1]

    lb_max = max(base_template.x[0], 0.8 * lb)
    ub_min = min(base_template.x[-1], 1.2 * ub)

    mask = (lb <= base_template.x) & (base_template.x <= ub)
    max_mask = (lb_max <= base_template.x) & (base_template.x <= ub_min)

    new_fwhm = base_template.fwhm[:1].copy()
    new_x = base_template.x[max_mask].copy()
    new_data = np.where(mask, base_template.data[0], 0.0)[max_mask][None, :].copy()

    new_template = IronTemplate(
        fwhm=new_fwhm,
        x=new_x,
        data=new_data,
        is_logspace=base_template.is_logspace,
        sigma_res=base_template.sigma_res,
        name=args.name,
        path=None,
        fwhm_norm=base_template.fwhm_norm,
        normalisation=None,
    )

    x_log = _get_xlog(
        (new_template.x[0], new_template.x[-1]),
        info.loading.sigma_res,
    )

    _template = new_template.createLogspace(x_log, inplace=False)
    _template.resample(base_template.fwhm, inplace=True)
    new_template.mimicLogspace(_template, inplace=True)

    if args.save:
        new_template.save_to_cache(info)

    plot_template(base_template, new_template, info)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a custom Iron template.")
    parser.add_argument(
        "base",
        type=str,
        help="Name of the base template.",
    )
    parser.add_argument(
        "name",
        type=str,
        help="Name of the new template (without .fits extension).",
    )
    parser.add_argument(
        "-r",
        "--v_res",
        type=float,
        help="Velocity resolution (km/s).",
        default=69.0,
    )
    parser.add_argument(
        "-l",
        "--lb",
        type=float,
        help="Lower bound of the wavelength range (Angstroms).",
        default=0.0,
    )
    parser.add_argument(
        "-u",
        "--ub",
        type=float,
        help="Upper bound of the wavelength range (Angstroms).",
        default=0.0,
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
