from collections import Counter
from math import isfinite

from astropy.modeling import Model, Parameter
from astropy.units import Unit
from h5py import File, Group, string_dtype
from numpy import bool_, dtype, empty, float32, int_

from ...modeling.utils import LinearTie
from ..utils import serialize_parameter


def _validate_same_classes(objs: list[object], _cls: type | None = None) -> None:
    if _cls is None:
        _cls = objs[0].__class__

    cls_counts = Counter(obj.__class__ for obj in objs)
    if _cls not in cls_counts:
        raise ValueError(f"Expected at least one instance of {_cls.__name__}!")
    if len(cls_counts) > 1:
        raise ValueError(f"Expected all objects to be instances of {_cls.__name__}!")

def _validate_same_parameter_names(parameters: list[Parameter]) -> None:
    names = [p.name for p in parameters]
    if len(set(names)) != 1:
        raise ValueError("All parameters must have the same name!")

###

def parameter_to_hdf5(
    parameter: Parameter,
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    unit: str = "",
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    """
    Save an astropy Parameter to an HDF5 group.
    """
    if group_name is None:
        group_name = parameter.name

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the file.")
        del f[group_name]

    grp = f.create_group(group_name)
    serialized = serialize_parameter(parameter, unit=unit)
    if serialized["unit"]:
        grp.attrs["unit"] = serialized["unit"]

    ds = grp.create_dataset(
        "value", 
        data=serialized["value"], 
        dtype=_dtype, 
        compression=compression, 
        compression_opts=compression_opts,
    )
    grp.create_dataset(
        "fixed",
        data=serialized["fixed"],
        dtype=bool_, 
        compression=compression, 
        compression_opts=compression_opts,
    )
    grp.create_dataset(
        "bounds",
        data=serialized["bounds"],
        shape=(2,),
        dtype=_dtype,
        compression=compression,
        compression_opts=compression_opts,
    )
    if serialized["tied"] is not None:
        ds = grp.create_dataset(
            "tied",
            data=(serialized["tied"]["a"], serialized["tied"]["b"]),
            shape=(2,),
            dtype=_dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        ds.attrs["model_name"] = serialized["tied"]["model_name"]
        ds.attrs["parameter_name"] = serialized["tied"]["parameter_name"]

    return grp

def parameters_to_hdf5(
    parameters: list[Parameter],
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    unit: str = "",
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    """
    Write a list of astropy Parameters to an HDF5 group.
    """
    _validate_same_parameter_names(parameters)

    if group_name is None:
        group_name = parameters[0].name

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the file.")
        del f[group_name]

    grp = f.create_group(group_name)

    n = len(parameters)
    values = empty((n,), dtype=_dtype)
    fixed = empty((n,), dtype=bool_)
    bounds = empty((n, 2), dtype=_dtype)

    tied_indices = []
    tied_abs = []
    tied_model_names = []
    tied_parameter_names = []

    for i, p in enumerate(parameters):
        serialized = serialize_parameter(p, unit=unit)
        
        values[i] = serialized["value"]
        fixed[i] = bool(serialized["fixed"])
        bounds[i,:] = serialized["bounds"]

        if serialized["tied"] is not None:
            tied_indices.append(i)
            tied_abs.append((serialized["tied"]["a"], serialized["tied"]["b"])) 
            tied_model_names.append(serialized["tied"]["model_name"])
            tied_parameter_names.append(serialized["tied"]["parameter_name"])

    if unit:
        grp.attrs["unit"] = unit

    # write datasets
    dt = string_dtype(encoding="utf-8")
    grp.create_dataset(
        "values", 
        data=values, 
        shape=(n,),
        dtype=_dtype,
        compression=compression, 
        compression_opts=compression_opts,
    )
    grp.create_dataset(
        "fixed", 
        data=fixed,
        shape=(n,),
        dtype=bool_,
        compression=compression, 
        compression_opts=compression_opts,
    )
    grp.create_dataset(
        "bounds", 
        data=bounds, 
        shape=(n, 2), 
        dtype=_dtype,
        compression=compression, 
        compression_opts=compression_opts,
    )
    if tied_indices:
        m = len(tied_indices)
        grp.create_dataset(
            "tied_indices",
            data=tied_indices,
            shape=(m,),
            dtype=int_,
            compression=compression,
            compression_opts=compression_opts,
        )
        grp.create_dataset(
            "tied_abs",
            data=tied_abs,
            shape=(m, 2),
            dtype=_dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        grp.create_dataset(
            "tied_model_names",
            data=tied_model_names,
            shape=(m,),
            dtype=dt,
            compression=compression,
            compression_opts=compression_opts,
        )
        grp.create_dataset(
            "tied_parameter_names",
            data=tied_parameter_names,
            shape=(m,),
            dtype=dt,
            compression=compression,
            compression_opts=compression_opts,
        )

    return grp

###

def model_to_hdf5(
    model: Model,
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    unit_map: dict[str, str] | None = None,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    """
    Save a model (collection of astropy Parameters) to an HDF5 group.

    The output structure contains only the model's parameters under:
    `{group_name}/parameters/{parameter_name}`
    """
    if group_name is None:
        group_name = model.name or "model"
    if unit_map is None:
        unit_map = {}

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the file.")
        del f[group_name]


    grp = f.create_group(group_name)

    # store model name on the model group for round-trip fidelity
    grp.attrs["model_name"] = model.name
    parameters = grp.create_group("parameters")
    for param_name in model.param_names:
        parameter_to_hdf5(
            getattr(model, param_name),
            parameters,
            group_name=param_name,
            _dtype=_dtype,
            unit=unit_map.get(param_name, ""),
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )

    return grp

def models_to_hdf5(
    models: list[Model],
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    unit_map: dict[str, str] | None = None,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    """
    Write a list of astropy Models to an HDF5 group.
    """
    _validate_same_classes(models)

    if group_name is None:
        group_name = models[0].__class__.__name__
    if unit_map is None:
        unit_map = {}

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the file.")
        del f[group_name]

    grp = f.create_group(group_name)
    grp.attrs["n_models"] = len(models)

    grp.create_dataset(
        "names", 
        data=[m.name for m in models],
        dtype=string_dtype(encoding='utf-8'),
    )
    parameters = grp.create_group("parameters")
    for param_name in models[0].param_names:
        parameters_to_hdf5(
            [getattr(m, param_name) for m in models],
            parameters,
            group_name=param_name,
            _dtype=_dtype,
            unit=unit_map.get(param_name, ""),
            force=force,
            compression=compression,
            compression_opts=compression_opts,
        )

    return grp

###

def hdf5_to_parameter(
    parameter: Parameter,
    grp: Group,
    unit: str = "",
) -> None:

    if unit and ("unit" in grp.attrs):
        def converter(val: float | None) -> float | None:
            if val is None:
                return None
            return (val * Unit(grp.attrs["unit"])).to(unit).value
    elif (not unit) and ("unit" not in grp.attrs):
        def converter(val: float | None) -> float | None:
            if val is None:
                return None
            return float(val)
    else:
        raise ValueError(f"Unit mismatch: expected '{unit}', attrs '{grp.attrs}'")
    
    parameter.value = converter(grp["value"][()])
    parameter.fixed = bool(grp["fixed"][()])
    
    if "bounds" in grp:
        parameter.bounds = tuple(
            b if isfinite(b) else None
            for b in map(converter, grp["bounds"][()])
        )
    else:
        parameter.bounds = (None, None)

    if "tied" in grp:
        parameter.tied = LinearTie(
            a=float(grp["tied"][0]),
            b=float(grp["tied"][1]),
            model_name=grp["tied"].attrs["model_name"],
            parameter_name=grp["tied"].attrs["parameter_name"],
        )
    else:
        parameter.tied = None

def hdf5_to_parameters(
    parameters: list[Parameter],
    grp: Group,
    unit: str = "",
) -> None:
    """Restore parameters from a compact HDF5 group created by
    ``parameters_to_hdf5``. Parameters are matched by name.
    """
    _validate_same_parameter_names(parameters)

    if unit and ("unit" in grp.attrs):
        def converter(val: float | None) -> float | None:
            if val is None:
                return None
            return (val * Unit(grp.attrs["unit"])).to(unit).value
    elif (not unit) and ("unit" not in grp.attrs):
        def converter(val: float | None) -> float | None:
            if val is None:
                return None
            return float(val)
    else:
        raise ValueError(f"Unit mismatch: expected '{unit}', attrs '{grp.attrs}'")

    values = grp["values"][()]
    fixed = grp["fixed"][()]
    bounds = grp["bounds"][()]

    if "tied_indices" in grp:
        tied_indices: list[int] = list(grp["tied_indices"][()])
        tied_abs = grp["tied_abs"][()]
        tied_model_names = grp["tied_model_names"].asstr()[()]
        tied_parameter_names = grp["tied_parameter_names"].asstr()[()]
    else:
        tied_indices = []

    for i, parameter in enumerate(parameters):
        parameter.value = converter(values[i])
        parameter.fixed = bool(fixed[i])
        parameter.bounds = tuple(
            b if isfinite(b) else None
            for b in map(converter, bounds[i])
        )

        if i in tied_indices:
            idx = tied_indices.index(i)
            parameter.tied = LinearTie(
                a=float(tied_abs[idx][0]),
                b=float(tied_abs[idx][1]),
                model_name=tied_model_names[idx],
                parameter_name=tied_parameter_names[idx],
            )
        else:
            parameter.tied = None


def hdf5_to_model(
    model: Model,
    grp: Group,
    unit_map: dict[str, str] | None = None,
) -> None:

    if unit_map is None:
        unit_map = {}

    # Use the group name as the model name if available; this keeps the
    # round-trip name stable when a specific group_name is provided by the
    # caller.
    # Prefer an explicit stored model_name attribute; otherwise fall back to
    # the HDF5 group's name.
    model.name = grp.attrs.get("model_name", grp.name.split("/")[-1])

    parameters_grp = grp["parameters"]
    for param_name in model.param_names:
        hdf5_to_parameter(
            getattr(model, param_name),
            parameters_grp[param_name],
            unit=unit_map.get(param_name, ""),
        )


def hdf5_to_models(
    models: list[Model],
    grp: Group,
    unit_map: dict[str, str] | None = None,
) -> None:
    """Restore an iterable of Models from a compact HDF5 group created by
    ``models_to_hdf5``. Models are matched by name and parameters by order.
    """
    _validate_same_classes(models)
    
    if unit_map is None:
        unit_map = {}

    for model, name in zip(models, grp["names"].asstr()[()]):
        model.name = name

    assert len(models) == grp.attrs["n_models"]

    parameters_grp = grp["parameters"]
    for param_name in models[0].param_names:
        hdf5_to_parameters(
            [getattr(m, param_name) for m in models],
            parameters_grp[param_name],
            unit=unit_map.get(param_name, ""),
        )