from h5py import File, Group
from numpy import dtype, float32
from quasar_utils.setup import Info

from ...continuum import PowerLawModel
from .utils import hdf5_to_model, model_to_hdf5


def powerlaw_to_hdf5(
    model: PowerLawModel, 
    f: File | Group,
    info: Info,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    unit_map: dict[str, str] = {
        "flux": str(info.units.flux_unit),
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
    return grp

def hdf5_to_powerlaw(
    grp: Group,
    info: Info,
) -> PowerLawModel:
    model = PowerLawModel()
    unit_map: dict[str, str] = {
        "flux": str(info.units.flux_unit),
    }
    hdf5_to_model(
        model,
        grp,
        unit_map=unit_map,
    )
    model.x0 = info.continuum.x0
    model.y0 = info.continuum.y0
    return model