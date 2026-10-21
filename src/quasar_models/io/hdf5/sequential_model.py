from collections.abc import Iterable
from pathlib import Path

from h5py import File, Group, string_dtype
from numpy import dtype, float32
from quasar_typing.astropy.modeling import CompoundModelType
from quasar_utils.setup import Info
from scipy.sparse import csr_matrix

from quasar_models.modeling.base_model import BaseModel
from quasar_models.modeling.sequential_model import SequentialModel

from ..json import file_to_model, model_to_file


def _validate(grp: Group, info: Info) -> None:
    """Validate a SequentialModel HDF5 group.

    1. Checks minimal expected datasets and basic shapes.
    2. Confirms that the UnitsInfo hashes match between the HDF5 group and the 
    provided Info object.
    
    Raises TypeError/ValueError with a clear message when validation fails.
    """
    if not isinstance(grp, Group):
        raise TypeError(
            "Expected an h5py.Group for sequential model validation"
        )

    hash_algo = grp.attrs.get("units_hash_algo", None)
    if hash_algo is None:
        raise ValueError(
            "Missing attribute: 'units_hash_algo' in sequential model group"
        )

    units_hash = grp.attrs.get("units_hash", None)
    if units_hash is None:
        raise ValueError(
            "Missing attribute: 'units_hash' in sequential model group"
        )

    my_units_hash = info.units.create_hash(hash_algo)
    if my_units_hash != units_hash:
        raise ValueError(f"UnitsInfo hash mismatch {hash_algo}")

    if "names" not in grp:
        raise ValueError("Missing dataset: 'names' in sequential model group")
    if "initial_values" not in grp:
        raise ValueError(
            "Missing dataset: 'initial_values' in sequential model group"
        )

    names = grp["names"].asstr()[()]
    if not isinstance(names, Iterable):
        raise TypeError("Dataset 'names' must be an iterable of strings")

    initial_values = grp["initial_values"][()]
    if initial_values.ndim != 1:
        raise ValueError(
            f"Dataset 'initial_values' must be 1D, got shape {initial_values.shape}"
        )

    if "kernels" in grp:
        kernels = grp["kernels"]
        if kernels.ndim != 2:
            raise ValueError(
                f"Dataset 'kernels' must be 2D, got shape {kernels.shape}"
            )


def sequential_model_to_hdf5(
    f: File | Group,
    seq_model: SequentialModel,
    info: Info,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    """
    Save a SequentialModel instance to a HDF5 group. 

    To save space, the following information is stored:
    - Names of the submodels: used to reconstruct the submodels from a parent 
    model, while respecting their original order.
    - Initial values: the starting parameters for the submodels.
    - Kernels: the kernel matrix used to correct for instrumental resolution, if 
    applicable.

    Parameters
    ----------
    seq_model : SequentialModel
    f : File | Group
    info : Info
    group_name : str | None
    _dtype : str | dtype
    force : bool
    compression : str | None
    compression_opts : int | None
    
    Returns
    -------
    grp : Group
        The HDF5 group containing the saved SequentialModel instance.
    """
    if group_name is None:
        group_name = seq_model.__class__.__name__

    if group_name in f:
        if not force:
            raise ValueError(
                f"Group '{group_name}' already exists in the HDF5 file."
            )
        del f[group_name]

    _model = seq_model._model
    ms = (_model,) if _model.n_submodels == 1 else tuple(_model)
    compression_kwargs = {
        "compression": compression,
        "compression_opts": compression_opts,
    }

    grp = f.create_group(group_name)
    grp.create_dataset(
        "names",
        data=[m.name for m in ms],
        dtype=string_dtype(encoding="utf-8"),
        **compression_kwargs,
    )
    grp.create_dataset(
        "initial_values",
        data=seq_model._initial_values,
        dtype=_dtype,
        **compression_kwargs,
    )
    if seq_model.kernels is not None:
        kernels = seq_model.kernels
        if isinstance(kernels, csr_matrix):
            kernels = kernels.toarray()

        grp.create_dataset(
            "kernels", data=kernels, dtype=_dtype, **compression_kwargs,
        )

    # Store UnitsInfo hash
    grp.attrs["units_hash_algo"] = "sha256"
    grp.attrs["units_hash"] = info.units.create_hash("sha256")

    _validate(grp, info)
    return grp


def hdf5_to_sequential_model(
    grp: Group,
    model: BaseModel | CompoundModelType[BaseModel],
    info: Info,
) -> SequentialModel:
    """
    Load a SequentialModel instance from a HDF5 group. 

    To save space, minimal information was saved to file:
    - Names of the submodels: used to reconstruct the submodels from a parent 
      model, while respecting their original order.
    - Initial values: the starting parameters for the submodels.
    - Kernels: the kernel matrix used to correct for instrumental resolution, if 
      applicable.

    Parameters
    ----------
    grp : Group
        The HDF5 group containing the saved SequentialModel instance.
    model : BaseModel, CompoundModelType[BaseModel]
        The parent model from which the submodels will be reconstructed.
    info : Info

    Returns
    -------
    SequentialModel
        The reconstructed SequentialModel instance.
    """
    _validate(grp, info)
    names: list[str] = list(grp["names"].asstr()[()])
    submodels: list[BaseModel] = [model[name].copy() for name in names]
    _model = sum(submodels[1:], start=submodels[0])
    _model.parameters = grp["initial_values"][()]
    kernels = grp["kernels"][()] if "kernels" in grp else None
    return SequentialModel(_model, kernels=kernels)


### Convenience functions


def dump_sequential_model(
    path: Path,
    seq_model: SequentialModel,
    info: Info,
    model_ext: str = "toon",
    force: bool = False,
    _dtype: str | dtype = float32,
    _name: str = "_sequential_model",
) -> Path:
    """
    Convenience function for saving a SequentialModel instance.
    """
    dir_path = path / _name
    dir_path.mkdir(parents=True, exist_ok=force)
    model_ext = model_ext.removeprefix(".").lower()

    model_path = dir_path / f"model.{model_ext}"
    model_to_file(model_path, seq_model._model, info)

    hdf5_path = dir_path / "sequential_model.h5"
    with File(hdf5_path, "w") as f:
        sequential_model_to_hdf5(
            f, seq_model, info, _dtype=_dtype, force=force
        )

    return dir_path


def load_sequential_model(
    dir_path: Path, info: Info, _name: str = "_sequential_model",
) -> SequentialModel:
    """
    Convenience function for loading a SequentialModel instance.

    Notes
    -----
    Unlike `file_to_model`, `hdf5_to_sequential_model` takes no Info instance as
    a parameter, as it assumes an equivalent Info instance was used to save the 
    model.
    """
    if dir_path.name != _name:
        dir_path /= _name

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path!s}")

    try:
        model_path = next(dir_path.glob("model.*"))
        model = file_to_model(model_path, info)
    except StopIteration:
        raise FileNotFoundError(
            f"No model file found in directory: {dir_path!s}"
        )

    hdf5_path = dir_path / "sequential_model.h5"
    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"Could not find sequential model HDF5 file: {hdf5_path!s}"
        )

    with File(hdf5_path, "r") as f:
        seq_model = hdf5_to_sequential_model(f, model, info)

    return seq_model
