from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datetime as _datetime
import json

import h5py
import numpy as np


HDF5_EXTENSIONS = {".h5", ".hdf5"}
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_RESERVED_GROUP_NAMES = {"axes", "outputs", "status"}


class MapOutputError(TypeError):
    """Raised when evaluate() returns outputs that are not scalar map values."""


@dataclass(frozen=True)
class Axis:
    """One independent variable for a generated map."""

    name: str
    values: np.ndarray
    units: str = ""
    spacing: str = "values"

    @classmethod
    def linear(
        cls,
        name: str,
        start: float,
        stop: float,
        count: int,
        units: str = "",
    ) -> "Axis":
        return cls(
            name=name,
            values=np.linspace(start, stop, count, dtype=float),
            units=units,
            spacing="linear",
        )

    @classmethod
    def log(
        cls,
        name: str,
        start: float,
        stop: float,
        count: int,
        units: str = "",
    ) -> "Axis":
        return cls(
            name=name,
            values=np.geomspace(start, stop, count, dtype=float),
            units=units,
            spacing="log",
        )

    @classmethod
    def values(
        cls,
        name: str,
        values,
        units: str = "",
    ) -> "Axis":
        return cls(
            name=name,
            values=np.asarray(values, dtype=float),
            units=units,
            spacing="values",
        )


def _hdf5_filename(filename: str | Path) -> str:
    path = Path(filename)

    if path.suffix == "":
        path = path.with_suffix(".h5")
    elif path.suffix.lower() not in HDF5_EXTENSIONS:
        raise ValueError("Map files must use .h5, .hdf5, or no extension.")

    path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


def _validate_hdf5_name(name: str, label: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} names must be non-empty strings.")

    if "/" in name:
        raise ValueError(f"{label} name '{name}' cannot contain '/'.")


def _validate_axis(axis: Axis) -> None:
    _validate_hdf5_name(axis.name, "Axis")

    values = np.asarray(axis.values, dtype=float)

    if axis.spacing not in {"linear", "log", "values"}:
        raise ValueError(f"Axis '{axis.name}' has unsupported spacing '{axis.spacing}'.")

    if values.ndim != 1:
        raise ValueError(f"Axis '{axis.name}' must be one-dimensional.")

    if len(values) < 2:
        raise ValueError(f"Axis '{axis.name}' requires at least two points.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Axis '{axis.name}' values must all be finite.")

    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"Axis '{axis.name}' must be strictly increasing.")

    if axis.spacing == "log" and np.any(values <= 0.0):
        raise ValueError(f"Log axis '{axis.name}' must contain only positive values.")


def _shape_from_axes(axes: list[Axis]) -> tuple[int, ...]:
    return tuple(len(axis.values) for axis in axes)


def _inputs_from_index(axes: list[Axis], index: tuple[int, ...], constants: dict) -> dict[str, float]:
    inputs = dict(constants)

    for axis, i in zip(axes, index):
        inputs[axis.name] = float(axis.values[i])

    return inputs


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return repr(value)


def _dataset_names(group, excluded: set[str]) -> list[str]:
    return [
        name
        for name, item in group.items()
        if name not in excluded and isinstance(item, h5py.Dataset)
    ]


def _validate_scalar_outputs(
    values: dict,
    outputs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    if not isinstance(values, dict):
        raise MapOutputError("evaluate() must return a dictionary of scalar outputs.")

    scalar_outputs: dict[str, float] = {}

    for output_name, value in values.items():
        try:
            _validate_hdf5_name(output_name, "Output")
        except ValueError as exc:
            raise MapOutputError(str(exc)) from exc

        if output_name in _RESERVED_GROUP_NAMES:
            raise MapOutputError(f"evaluate() output name '{output_name}' is reserved.")

        if isinstance(value, dict):
            raise MapOutputError(
                f"evaluate() output '{output_name}' must be a scalar number. "
                "Nested dictionaries and vector outputs are not supported. "
                "Return individual scalar entries instead."
            )

        if isinstance(value, (str, bytes, bool, np.bool_)):
            raise MapOutputError(f"evaluate() output '{output_name}' must be numeric.")

        array = np.asarray(value)

        if array.shape != ():
            raise MapOutputError(
                f"evaluate() output '{output_name}' must be a scalar number. "
                "Lists, tuples, arrays, and vector outputs are not supported."
            )

        try:
            scalar_outputs[output_name] = float(array)
        except Exception as exc:
            raise MapOutputError(f"evaluate() output '{output_name}' must be numeric.") from exc

    if outputs is not None:
        missing = [
            output_name
            for output_name in outputs
            if output_name not in scalar_outputs
        ]

        if missing:
            raise MapOutputError(f"evaluate() did not return required outputs: {missing}")

        scalar_outputs = {
            output_name: scalar_outputs[output_name]
            for output_name in outputs
        }

    return scalar_outputs


def _write_axis_dataset(axes_group: h5py.Group, axis: Axis) -> None:
    dataset = axes_group.create_dataset(
        axis.name,
        data=np.asarray(axis.values, dtype=float),
    )

    dataset.attrs["name"] = axis.name
    dataset.attrs["units"] = axis.units
    dataset.attrs["spacing"] = axis.spacing


def _create_group(
    file: h5py.File,
    group: str,
    axes: list[Axis],
    output_names: list[str],
    constants: dict,
    metadata: dict,
    fill_value: float,
    compression: str | None,
    compression_opts,
) -> h5py.Group:
    map_group = file.create_group(group)

    map_group.attrs["map_format"] = "fullflow-map-v2"
    map_group.attrs["created_utc"] = _datetime.datetime.now(_datetime.UTC).isoformat()
    map_group.attrs["axis_order"] = json.dumps([axis.name for axis in axes])
    map_group.attrs["output_names"] = json.dumps(output_names)
    map_group.attrs["constants"] = json.dumps(constants, default=_json_default)
    map_group.attrs["metadata"] = json.dumps(metadata, default=_json_default)

    axes_group = map_group.create_group("axes")
    outputs_group = map_group.create_group("outputs")
    status_group = map_group.create_group("status")

    for axis in axes:
        _write_axis_dataset(axes_group, axis)

    shape = _shape_from_axes(axes)

    if compression is None:
        compression_opts = None

    for output_name in output_names:
        outputs_group.create_dataset(
            output_name,
            shape=shape,
            dtype="f8",
            chunks=True,
            compression=compression,
            compression_opts=compression_opts,
            fillvalue=fill_value,
        )

    status_group.create_dataset(
        "success",
        shape=shape,
        dtype="?",
        chunks=True,
        compression=compression,
        compression_opts=compression_opts,
        fillvalue=False,
    )

    status_group.create_dataset(
        "message",
        shape=shape,
        dtype=_STRING_DTYPE,
        chunks=True,
    )

    return map_group


def _check_existing_group(
    map_group: h5py.Group,
    axes: list[Axis],
    outputs: list[str] | tuple[str, ...] | None,
) -> list[str]:
    shape = _shape_from_axes(axes)

    if "axes" not in map_group or not isinstance(map_group["axes"], h5py.Group):
        raise ValueError("Existing map group is not a fullflow-map-v2 group and cannot be resumed.")

    if "outputs" not in map_group or not isinstance(map_group["outputs"], h5py.Group):
        raise ValueError("Existing map group is missing required group 'outputs'.")

    if "status" not in map_group or not isinstance(map_group["status"], h5py.Group):
        raise ValueError("Existing map group is missing required group 'status'.")

    axes_group = map_group["axes"]
    outputs_group = map_group["outputs"]
    status_group = map_group["status"]

    for axis in axes:
        if axis.name not in axes_group:
            raise ValueError(f"Existing map group is missing axis '{axis.name}'.")

        axis_values = np.asarray(axes_group[axis.name][()], dtype=float)

        if not np.array_equal(axis_values, np.asarray(axis.values, dtype=float)):
            raise ValueError(f"Existing map axis '{axis.name}' does not match requested axis.")

    axis_order = json.loads(map_group.attrs.get("axis_order", "[]"))

    if axis_order and axis_order != [axis.name for axis in axes]:
        raise ValueError("Existing map axis order does not match requested axis order.")

    output_names = list(outputs) if outputs is not None else _dataset_names(outputs_group, set())

    if not output_names:
        raise ValueError("Existing map group does not contain any output datasets.")

    missing = [
        output_name
        for output_name in output_names
        if output_name not in outputs_group
    ]

    if missing:
        raise ValueError(f"Existing map group is missing requested outputs: {missing}")

    for output_name in output_names:
        if outputs_group[output_name].shape != shape:
            raise ValueError(
                f"Existing output '{output_name}' has shape {outputs_group[output_name].shape}; "
                f"expected {shape}."
            )

    if "success" not in status_group:
        raise ValueError("Existing map group is missing required dataset 'status/success'.")

    if "message" not in status_group:
        raise ValueError("Existing map group is missing required dataset 'status/message'.")

    if status_group["success"].shape != shape:
        raise ValueError("Existing status/success shape does not match requested map axes.")

    return output_names


def _discover_outputs(
    axes: list[Axis],
    constants: dict,
    evaluate: Callable,
    outputs: list[str] | tuple[str, ...] | None,
    raise_errors: bool,
) -> tuple[list[str], tuple[int, ...], dict[str, float], list[tuple[tuple[int, ...], str]]]:
    shape = _shape_from_axes(axes)
    failures: list[tuple[tuple[int, ...], str]] = []

    for index in np.ndindex(shape):
        inputs = _inputs_from_index(axes, index, constants)

        try:
            values = _validate_scalar_outputs(evaluate(**inputs), outputs)
            return list(values.keys()), index, values, failures
        except MapOutputError:
            raise
        except Exception as exc:
            if raise_errors:
                raise

            failures.append((index, f"{type(exc).__name__}: {exc}"))

    raise RuntimeError("Could not discover map outputs because every evaluated point failed.")


def _write_failure(
    map_group: h5py.Group,
    index: tuple[int, ...],
    output_names: list[str],
    fill_value: float,
    failure_message: str,
) -> None:
    outputs_group = map_group["outputs"]
    status_group = map_group["status"]

    for output_name in output_names:
        outputs_group[output_name][index] = fill_value

    status_group["success"][index] = False
    status_group["message"][index] = failure_message


def generate_map(
    filename: str | Path,
    axes: list[Axis] | tuple[Axis, ...],
    evaluate: Callable,
    group: str = "map",
    outputs: list[str] | tuple[str, ...] | None = None,
    constants: dict | None = None,
    metadata: dict | None = None,
    resume: bool = True,
    overwrite: bool = False,
    fill_value: float = np.nan,
    compression: str | None = "gzip",
    compression_opts=None,
    flush_every: int = 25,
    raise_errors: bool = False,
) -> str:
    """Generate a FullFlow-compatible N-dimensional scalar HDF5 map.

    ``evaluate`` is called as ``evaluate(**inputs)`` where ``inputs`` contains
    one value for each axis plus any constants. It must return a flat
    ``dict[str, scalar_number]``. Vector outputs are intentionally rejected;
    return individual scalar values instead, such as ``Y_H2O`` or ``X_CO2``.
    """
    filename = _hdf5_filename(filename)
    axes = list(axes)
    constants = {} if constants is None else dict(constants)
    metadata = {} if metadata is None else dict(metadata)

    if not axes:
        raise ValueError("At least one axis is required.")

    axis_names = [axis.name for axis in axes]

    if len(axis_names) != len(set(axis_names)):
        raise ValueError("Axis names must be unique.")

    if set(axis_names) & set(constants):
        raise ValueError("Axis names cannot also be used as constant input names.")

    for axis in axes:
        _validate_axis(axis)

    if outputs is not None:
        outputs = list(outputs)

        if not outputs:
            raise ValueError("outputs must contain at least one output name when provided.")

        if len(outputs) != len(set(outputs)):
            raise ValueError("Output names must be unique.")

        for output_name in outputs:
            _validate_hdf5_name(output_name, "Output")

            if output_name in _RESERVED_GROUP_NAMES:
                raise ValueError(f"Output name '{output_name}' is reserved.")

    group = group.strip("/") or "map"

    with h5py.File(filename, "a") as file:
        if group in file:
            if overwrite:
                del file[group]
            elif not resume:
                raise FileExistsError(
                    f"HDF5 group '{group}' already exists in '{filename}'. "
                    "Use resume=True or overwrite=True."
                )

        if group in file:
            map_group = file[group]
            output_names = _check_existing_group(map_group, axes, outputs)

        else:
            if outputs is None:
                output_names, first_index, first_values, failures = _discover_outputs(
                    axes,
                    constants,
                    evaluate,
                    outputs,
                    raise_errors,
                )
            else:
                output_names = list(outputs)
                first_index = None
                first_values = None
                failures = []

            map_group = _create_group(
                file,
                group,
                axes,
                output_names,
                constants,
                metadata,
                fill_value,
                compression,
                compression_opts,
            )

            for index, failure_message in failures:
                _write_failure(map_group, index, output_names, fill_value, failure_message)

            if first_index is not None and first_values is not None:
                for output_name, value in first_values.items():
                    map_group["outputs"][output_name][first_index] = value

                map_group["status/success"][first_index] = True
                map_group["status/message"][first_index] = ""

        shape = _shape_from_axes(axes)
        status_group = map_group["status"]
        outputs_group = map_group["outputs"]
        counter = 0

        for index in np.ndindex(shape):
            if resume and bool(status_group["success"][index]):
                continue

            inputs = _inputs_from_index(axes, index, constants)

            try:
                values = _validate_scalar_outputs(evaluate(**inputs), output_names)

                for output_name, value in values.items():
                    outputs_group[output_name][index] = value

                status_group["success"][index] = True
                status_group["message"][index] = ""

            except MapOutputError:
                raise
            except Exception as exc:
                if raise_errors:
                    raise

                _write_failure(
                    map_group,
                    index,
                    output_names,
                    fill_value,
                    f"{type(exc).__name__}: {exc}",
                )

            counter += 1

            if flush_every and counter % flush_every == 0:
                file.flush()

        file.flush()

    return filename
