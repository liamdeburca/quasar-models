from astropy.units import Unit
from h5py import File, Group, string_dtype
from numpy import dtype, float32
from quasar_utils.setup import Info

from quasar_models.line import GaussianModel

from .utils import hdf5_to_model, hdf5_to_models, model_to_hdf5, models_to_hdf5


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


def gaussians_to_hdf5(
    models: list[GaussianModel],
    f: File | Group,
    info: Info,
    group_name: str = "gaussians",
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
    grp = models_to_hdf5(
        models,
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
    grp.create_dataset(
        "waves",
        data=[model.wave for model in models],
        dtype=_dtype,
        compression=compression,
        compression_opts=compression_opts,
    )
    grp.create_dataset(
        "linetypes",
        data=[model.linetype for model in models],
        dtype=string_dtype(),
        compression=compression,
        compression_opts=compression_opts,
    )
    grp.attrs["wavelength_unit"] = str(info.units.wavelength_unit)
    return grp


###


def hdf5_to_gaussian(grp: Group, info: Info,) -> GaussianModel:
    model = GaussianModel()
    unit_map: dict[str, str] = {
        "strength": str(info.units.strength_unit),
        "fwhm_v": "km/s",
        "v_off": "km/s",
    }
    hdf5_to_model(
        model, grp, unit_map=unit_map,
    )
    # Read metadata stored as explicit attributes
    model.wave = (
        (grp.attrs["wave"] * Unit(grp.attrs["wavelength_unit"]))
        .to(info.units.wavelength_unit)
        .value
    )
    model.sigma_res = info.loading.sigma_res
    model.linetype = grp.attrs["linetype"]
    model.n_sigmas = 3.0
    return model


def hdf5_to_gaussians(grp: Group, info: Info,) -> list[GaussianModel]:
    unit_map: dict[str, str] = {
        "strength": str(info.units.strength_unit),
        "fwhm_v": "km/s",
        "v_off": "km/s",
    }

    n_models = grp.attrs["n_models"]
    models = [GaussianModel() for _ in range(n_models)]
    waves = grp["waves"][()]
    linetypes = grp["linetypes"].asstr()[()]

    hdf5_to_models(models, grp, unit_map=unit_map)
    for model, wave, linetype in zip(models, waves, linetypes):
        model.wave = (
            (wave * Unit(grp.attrs["wavelength_unit"]))
            .to(info.units.wavelength_unit)
            .value
        )
        model.sigma_res = info.loading.sigma_res
        model.linetype = linetype
        model.n_sigmas = 3.0

    return models
