from astropy.units import Unit
from h5py import File, Group
from numpy import dtype, float32
from quasar_utils.setup import Info

from ...line import GaussianModel
from .utils import hdf5_to_model, model_to_hdf5


def gaussian_to_hdf5(
    model: GaussianModel, 
    f: File | Group,
    info: Info,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    unit_map: dict[str, str] = {
        "strength": str(info.units.strength_unit),
        "fwhm_v": "km/s",
        "v_off": "km/s",
    }
    grp = model_to_hdf5(
        model,
        f,
        group_name=group_name,
        _dtype=_dtype,
        unit_map=unit_map,
        force=force,
        compression=compression,
        compression_opts=compression_opts,
    )
    # Store metadata as explicit attributes. Avoid writing a Python dict
    # directly to HDF5 attributes which can lead to object-dtype storage
    # and is not portable across h5py versions.
    grp.attrs["wave"] = float(model.wave)
    grp.attrs["linetype"] = model.linetype
    grp.attrs["wavelength_unit"] = str(info.units.wavelength_unit)
    return grp


def hdf5_to_gaussian(
    grp: Group,
    info: Info,
) -> GaussianModel:
    model = GaussianModel()
    unit_map: dict[str, str] = {
        "strength": str(info.units.strength_unit),
        "fwhm_v": "km/s",
        "v_off": "km/s",
    }
    hdf5_to_model(
        model,
        grp,
        unit_map=unit_map,
    )
    # Read metadata stored as explicit attributes
    model.wave = (
        grp.attrs["wave"] * Unit(grp.attrs["wavelength_unit"])
    ).to(info.units.wavelength_unit).value
    model.sigma_res = info.loading.sigma_res
    model.linetype = grp.attrs["linetype"]
    model.n_sigmas = 3.0
    return model
