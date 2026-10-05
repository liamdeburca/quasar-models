from math import isfinite

from astropy.modeling import Model, Parameter
from astropy.units import Unit
from h5py import File, Group
from h5py import string_dtype
import numpy as _np
from numpy import bool_, dtype, float32

from ...modeling.utils import LinearTie
from ..utils import serialize_parameter


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


def parameters_to_hdf5(
    parameters: list[Parameter],
    f: File | Group,
    group_name: str | None = None,
    _dtype: str | dtype = float32,
    unit_map: dict[str, str] | None = None,
    force: bool = False,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> Group:
    """Write an iterable of Parameters efficiently into a single HDF5 group.

    The output layout under ``group_name`` contains compact arrays:
      - names: (N,) string dataset of parameter names
      - values: (N,) float dataset
      - fixed: (N,) bool dataset
      - bounds: (N,2) float dataset
      - units: (N,) string dataset (empty string if no unit)
      - tied: subgroup with arrays for tied entries (optional)
    """
    if group_name is None:
        group_name = "parameters"
    if unit_map is None:
        unit_map = {}

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the file.")
        del f[group_name]

    grp = f.create_group(group_name)

    params = list(parameters)
    n = len(params)

    names = [p.name for p in params]
    values = _np.empty((n,), dtype=_np.float64)
    fixed = _np.empty((n,), dtype=_np.bool_)
    bounds = _np.empty((n, 2), dtype=_np.float64)
    units = [""] * n

    tied_indices = []
    tied_a = []
    tied_b = []
    tied_model_names = []
    tied_parameter_names = []

    for i, p in enumerate(params):
        ser = serialize_parameter(p, unit=unit_map.get(p.name, ""))
        values[i] = ser["value"] if ser["value"] is not None else _np.nan
        fixed[i] = bool(ser["fixed"])
        lb, ub = ser["bounds"]
        bounds[i, 0] = lb if lb is not None else _np.nan
        bounds[i, 1] = ub if ub is not None else _np.nan
        units[i] = ser["unit"] or ""

        if ser["tied"] is not None:
            tied_indices.append(i)
            tied_a.append(ser["tied"]["a"]) 
            tied_b.append(ser["tied"]["b"]) 
            tied_model_names.append(ser["tied"]["model_name"])
            tied_parameter_names.append(ser["tied"]["parameter_name"])

    # write datasets
    dt = string_dtype(encoding="utf-8")
    grp.create_dataset("names", data=_np.array(names, dtype=dt), dtype=dt)
    grp.create_dataset("values", data=values.astype(_dtype), dtype=_dtype,
                       compression=compression, compression_opts=compression_opts)
    grp.create_dataset("fixed", data=fixed.astype(bool_), dtype=bool_,
                       compression=compression, compression_opts=compression_opts)
    grp.create_dataset("bounds", data=bounds.astype(_dtype), shape=(n, 2), dtype=_dtype,
                       compression=compression, compression_opts=compression_opts)
    grp.create_dataset("units", data=_np.array(units, dtype=dt), dtype=dt)

    if tied_indices:
        tgrp = grp.create_group("tied")
        tgrp.create_dataset("indices", data=_np.array(tied_indices, dtype=_np.int64))
        tgrp.create_dataset("a", data=_np.array(tied_a, dtype=_dtype), dtype=_dtype)
        tgrp.create_dataset("b", data=_np.array(tied_b, dtype=_dtype), dtype=_dtype)
        tgrp.create_dataset("model_names", data=_np.array(tied_model_names, dtype=dt), dtype=dt)
        tgrp.create_dataset("parameter_names", data=_np.array(tied_parameter_names, dtype=dt), dtype=dt)

    return grp


def hdf5_to_parameters(
    parameters: list[Parameter],
    grp: Group,
    unit_map: dict[str, str] | None = None,
) -> None:
    """Restore parameters from a compact HDF5 group created by
    ``parameters_to_hdf5``. Parameters are matched by name.
    """
    if unit_map is None:
        unit_map = {}

    params = {p.name: p for p in parameters}

    names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in grp['names'][()]]
    values = grp['values'][()]
    fixed = grp['fixed'][()]
    bounds = grp['bounds'][()]
    units = [u.decode('utf-8') if isinstance(u, bytes) else str(u) for u in grp['units'][()]]

    for i, name in enumerate(names):
        if name not in params:
            # skip unknown parameter (caller may supply subset)
            continue
        param = params[name]
        stored_unit = units[i] or None
        desired_unit = unit_map.get(name, "")

        # converter logic mirrors hdf5_to_parameter
        if desired_unit and stored_unit:
            def converter(v):
                if v is None or (_np.isnan(v) if isinstance(v, float) else False):
                    return None
                return (v * Unit(stored_unit)).to(desired_unit).value
        elif (not desired_unit) and (not stored_unit):
            def converter(v):
                if v is None or (_np.isnan(v) if isinstance(v, float) else False):
                    return None
                return float(v)
        else:
            raise ValueError(f"Unit mismatch: expected '{desired_unit}', attrs '{stored_unit}'")

        val = values[i]
        param.value = converter(val)
        param.fixed = bool(fixed[i])

        lb = bounds[i, 0]
        ub = bounds[i, 1]
        param.bounds = (
            None if (_np.isnan(lb) or not isfinite(lb)) else converter(lb),
            None if (_np.isnan(ub) or not isfinite(ub)) else converter(ub),
        )

    # process tied if present
    if 'tied' in grp:
        tgrp = grp['tied']
        indices = tgrp['indices'][()]
        a = tgrp['a'][()]
        b = tgrp['b'][()]
        model_names = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in tgrp['model_names'][()]]
        parameter_names = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in tgrp['parameter_names'][()]]

        for idx, aa, bb, mn, pn in zip(indices, a, b, model_names, parameter_names):
            name = names[idx]
            if name not in params:
                continue
            params[name].tied = LinearTie(a=float(aa), b=float(bb), model_name=mn, parameter_name=pn)


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
    """Write an iterable of astropy Models efficiently to a single HDF5 group.

    The function flattens all parameters across the provided models and stores
    concatenated arrays and an index map to reconstruct per-model slices.
    """
    if group_name is None:
        group_name = "models"
    if unit_map is None:
        unit_map = {}

    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the file.")
        del f[group_name]

    grp = f.create_group(group_name)

    models_list = list(models)
    model_names = [m.name for m in models_list]
    param_names = []
    values = []
    fixed = []
    bounds = []
    units = []
    model_index = []

    tied_indices = []
    tied_a = []
    tied_b = []
    tied_model_names = []
    tied_parameter_names = []

    for mi, m in enumerate(models_list):
        for pname in m.param_names:
            p = getattr(m, pname)
            ser = serialize_parameter(p, unit=unit_map.get(pname, unit_map.get(f"{m.name}.{pname}", "")))
            param_names.append(pname)
            values.append(ser['value'] if ser['value'] is not None else _np.nan)
            fixed.append(bool(ser['fixed']))
            lb, ub = ser['bounds']
            bounds.append((lb if lb is not None else _np.nan, ub if ub is not None else _np.nan))
            units.append(ser['unit'] or "")
            model_index.append(mi)

            if ser['tied'] is not None:
                tied_indices.append(len(values) - 1)
                tied_a.append(ser['tied']['a'])
                tied_b.append(ser['tied']['b'])
                tied_model_names.append(ser['tied']['model_name'])
                tied_parameter_names.append(ser['tied']['parameter_name'])

    n = len(values)
    dt = string_dtype(encoding='utf-8')
    grp.create_dataset('model_names', data=_np.array(model_names, dtype=dt), dtype=dt)
    grp.create_dataset('param_names', data=_np.array(param_names, dtype=dt), dtype=dt)
    grp.create_dataset('model_index', data=_np.array(model_index, dtype=_np.int64))
    grp.create_dataset('values', data=_np.array(values, dtype=_dtype), dtype=_dtype,
                       compression=compression, compression_opts=compression_opts)
    grp.create_dataset('fixed', data=_np.array(fixed, dtype=bool_), dtype=bool_,
                       compression=compression, compression_opts=compression_opts)
    grp.create_dataset('bounds', data=_np.array(bounds, dtype=_dtype), shape=(n, 2), dtype=_dtype,
                       compression=compression, compression_opts=compression_opts)
    grp.create_dataset('units', data=_np.array(units, dtype=dt), dtype=dt)

    if tied_indices:
        tgrp = grp.create_group('tied')
        tgrp.create_dataset('indices', data=_np.array(tied_indices, dtype=_np.int64))
        tgrp.create_dataset('a', data=_np.array(tied_a, dtype=_dtype), dtype=_dtype)
        tgrp.create_dataset('b', data=_np.array(tied_b, dtype=_dtype), dtype=_dtype)
        tgrp.create_dataset('model_names', data=_np.array(tied_model_names, dtype=dt), dtype=dt)
        tgrp.create_dataset('parameter_names', data=_np.array(tied_parameter_names, dtype=dt), dtype=dt)

    return grp


def hdf5_to_models(
    models: list[Model],
    grp: Group,
    unit_map: dict[str, str] | None = None,
) -> None:
    """Restore an iterable of Models from a compact HDF5 group created by
    ``models_to_hdf5``. Models are matched by name and parameters by order.
    """
    if unit_map is None:
        unit_map = {}

    models_map = {m.name: m for m in models}

    model_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in grp['model_names'][()]]
    param_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in grp['param_names'][()]]
    model_index = grp['model_index'][()]
    values = grp['values'][()]
    fixed = grp['fixed'][()]
    bounds = grp['bounds'][()]
    units = [u.decode('utf-8') if isinstance(u, bytes) else str(u) for u in grp['units'][()]]

    # Assign parameters by iterating over flattened arrays
    for idx, (m_idx, pname, val, fx, bds, stored_unit) in enumerate(
        zip(model_index, param_names, values, fixed, bounds, units)
    ):
        mname = model_names[m_idx]
        model = models_map.get(mname)
        if model is None:
            continue
        # find the matching Parameter on the model: match by occurrence/order
        # We select the next parameter on the model with name == pname that is
        # still unset. Simpler: use getattr to fetch it.
        if not hasattr(model, pname):
            continue
        param = getattr(model, pname)

        # determine desired unit: check both pname and model.pname key
        desired_unit = unit_map.get(pname, unit_map.get(f"{mname}.{pname}", ""))

        if desired_unit and stored_unit:
            def converter(v):
                if v is None or (_np.isnan(v) if isinstance(v, float) else False):
                    return None
                return (v * Unit(stored_unit)).to(desired_unit).value
        elif (not desired_unit) and (not stored_unit):
            def converter(v):
                if v is None or (_np.isnan(v) if isinstance(v, float) else False):
                    return None
                return float(v)
        else:
            raise ValueError(f"Unit mismatch: expected '{desired_unit}', attrs '{stored_unit}'")

        param.value = converter(val)
        param.fixed = bool(fx)
        lb, ub = bds
        param.bounds = (
            None if (_np.isnan(lb) or not isfinite(lb)) else converter(lb),
            None if (_np.isnan(ub) or not isfinite(ub)) else converter(ub),
        )

    # tied
    if 'tied' in grp:
        tgrp = grp['tied']
        indices = tgrp['indices'][()]
        a = tgrp['a'][()]
        b = tgrp['b'][()]
        t_model_names = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in tgrp['model_names'][()]]
        t_parameter_names = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in tgrp['parameter_names'][()]]

        for idx, aa, bb, mn, pn in zip(indices, a, b, t_model_names, t_parameter_names):
            # find parameter by flat index
            m_idx = int(model_index[idx])
            mname = model_names[m_idx]
            model = models_map.get(mname)
            if model is None:
                continue
            pname = param_names[idx]
            if not hasattr(model, pname):
                continue
            getattr(model, pname).tied = LinearTie(a=float(aa), b=float(bb), model_name=mn, parameter_name=pn)


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
