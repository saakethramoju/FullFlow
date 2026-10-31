"""Save linearization results to FullFlow HDF5 files."""

from __future__ import annotations

import h5py
import numpy as np

from fullflow.Exports.HDF5 import _initialize_file, hdf5_path, safe_group_name


def _path(result, network_name, solver_type):
    """Return the HDF5 path for a linearization result."""
    base = f"{safe_group_name(network_name)}/linearization/{solver_type}/runs"
    selected = [(name, option) for name, option in result.model_options if option is not None]
    if len(selected) == 1:
        name, option = selected[0]
        return f"{base}/{safe_group_name(name)}/{safe_group_name(option)}"
    return f"{base}/base"


def _write(group, result):
    """Write one linearization result into an HDF5 group."""
    strings = h5py.string_dtype(encoding="utf-8")
    for name in ("A", "B", "C", "D", "x0", "xdot0", "u0", "y0"):
        group.create_dataset(name, data=getattr(result, name))
    for name in ("states", "inputs", "outputs"):
        group.create_dataset(name, data=np.asarray(getattr(result, name), dtype=strings))
    group.create_dataset("time", data=float(result.time))
    options = group.create_group("model_options")
    for index, (name, option) in enumerate(result.model_options):
        item = options.create_group(str(index))
        item.attrs["name"] = name
        item.attrs["option"] = option if option is not None else ""
    diagnostics = group.create_group("diagnostics")
    for name, value in result.diagnostics.items():
        diagnostics.attrs[name] = value
    group.attrs["solver_type"] = result.solver_type


def save_linearization(result, filename, network_name=None, solver_type=None):
    """Save one linearization result to HDF5."""
    if not str(filename).strip():
        raise ValueError("filename must not be empty.")
    network_name = network_name or result.network_name
    solver_type = solver_type or result.solver_type
    path = hdf5_path(filename)
    group_path = _path(result, network_name, solver_type)
    with h5py.File(path, "a") as h5:
        _initialize_file(h5)
        group = h5.require_group(group_path)
        for name in list(group):
            if name != "points":
                del group[name]
        for name in list(group.attrs):
            del group.attrs[name]
        _write(group, result)
    return path


def save_linearizations(results, filename, network_name, solver_type="transient"):
    """Save time-indexed linearization results to HDF5."""
    if not str(filename).strip():
        raise ValueError("filename must not be empty.")
    path = hdf5_path(filename)
    with h5py.File(path, "a") as h5:
        _initialize_file(h5)
        group_paths = {_path(result, network_name, solver_type) for result in results.values()}
        for group_path in group_paths:
            points_path = f"{group_path}/points"
            if points_path in h5:
                del h5[points_path]
        for index, result in enumerate(results.values()):
            group_path = _path(result, network_name, solver_type)
            point = h5.require_group(f"{group_path}/points/{index}")
            _write(point, result)
    return path
