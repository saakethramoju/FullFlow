"""Simple HDF5 export helpers for FullFlow.

FullFlow HDF5 files are object containers.  Each top-level group is a named
FullFlow object, usually a Network or a generated Map.  Network solution data
lives under the network group, and map data lives directly under the map group.

Canonical layout
----------------

Network object::

    /<network_name>
        attrs: kind="network", name="original network name"
        /steady_state
            /components/<component>/<attribute>
            /table/<column>
            /diagnostics/<column>
            /statistics/<table>/<column>
        /transient
            /time
            /components/<component>/<attribute>
            /tracks/<tracked_name>
            /table/<column>
            /diagnostics/<column>
            /final/components/<component>/<attribute>
            /final/table/<column>

Map object::

    /<map_name>
        attrs: kind="map", name="original map name", axis_order=[...]
        /axes/<axis_name>
        /outputs/<output_name>
        /status/success
        /status/message
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any
import datetime as _datetime
import json
import math
import re

import h5py
import numpy as np


HDF5_EXTENSIONS = {".h5", ".hdf5"}
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_FORMAT = "fullflow-simple-hdf5"
_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class HDF5Target:
    """Internal target describing a group inside one HDF5 export file."""

    filename: str | Path
    group_path: str


def hdf5_path(filename: str | Path) -> Path:
    """Return an HDF5 path, adding ``.h5`` when the suffix is omitted."""
    path = Path(filename)

    if path.suffix == "":
        path = path.with_suffix(".h5")
    elif path.suffix.lower() not in HDF5_EXTENSIONS:
        raise ValueError(
            "FullFlow only exports HDF5 files. Use a filename with no extension, "
            ".h5, or .hdf5."
        )

    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def hdf5_filename(filename: str | Path) -> str:
    """Return an HDF5 filename string, adding ``.h5`` when omitted."""
    return str(hdf5_path(filename))


def dataset_names(group: h5py.Group, excluded: set[str] | None = None) -> list[str]:
    """Return direct child dataset names, excluding any requested names."""
    excluded = set() if excluded is None else set(excluded)
    return [
        name
        for name, item in group.items()
        if name not in excluded and isinstance(item, h5py.Dataset)
    ]


def hdf5_target(filename: str | Path | HDF5Target, group_path: str) -> HDF5Target:
    """Return an export target, preserving an existing target filename."""
    if isinstance(filename, HDF5Target):
        return HDF5Target(filename.filename, group_path)
    return HDF5Target(filename, group_path)


def safe_group_name(name: Any) -> str:
    """Return a compact HDF5-safe group/dataset name."""
    text = str(name).strip()
    text = re.sub(r"[\\/\x00]", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unnamed"


def object_path(name: Any) -> str:
    """Return the top-level group path for a named FullFlow object."""
    return "/" + safe_group_name(name)


def _now() -> str:
    return _datetime.datetime.now(_datetime.UTC).isoformat()


def _initialize_file(h5: h5py.File) -> None:
    h5.attrs["fullflow_format"] = _FORMAT
    h5.attrs["fullflow_schema_version"] = _SCHEMA_VERSION
    if "created_utc" not in h5.attrs:
        h5.attrs["created_utc"] = _now()
    h5.attrs["updated_utc"] = _now()


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(_plain(key)): _plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _as_text(value: Any) -> str:
    value = _plain(value)

    if isinstance(value, str):
        return value

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)

    if value is None:
        return ""

    return str(value)


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return math.nan
    return number


def _is_numeric_column(values: list[Any]) -> bool:
    if not values:
        return False

    for value in values:
        if value is None or isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, Real):
            continue
        try:
            float(value)
        except Exception:
            return False

    return True


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for column in row:
            column = str(column)
            if column in seen:
                continue
            seen.add(column)
            columns.append(column)

    return columns


def _replace_group(parent: h5py.Group, name: str) -> h5py.Group:
    name = safe_group_name(name)
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _require_object_group(h5: h5py.File, name: str, kind: str) -> h5py.Group:
    group = h5.require_group(safe_group_name(name))
    group.attrs["kind"] = kind
    group.attrs["name"] = str(name)
    group.attrs["updated_utc"] = _now()
    if "created_utc" not in group.attrs:
        group.attrs["created_utc"] = group.attrs["updated_utc"]
    return group


def _delete_children(group: h5py.Group, names: list[str]) -> None:
    for name in names:
        if name in group:
            del group[name]


def _write_dataset(parent: h5py.Group, name: str, value: Any, attrs: dict[str, Any] | None = None) -> h5py.Dataset:
    name = safe_group_name(name)
    if name in parent:
        del parent[name]

    value = _plain(value)
    attrs = {} if attrs is None else dict(attrs)

    if isinstance(value, (str, dict, list)) and not isinstance(value, (list, tuple)):
        dataset = parent.create_dataset(name, data=_as_text(value), dtype=_STRING_DTYPE)
    elif value is None:
        dataset = parent.create_dataset(name, data="", dtype=_STRING_DTYPE)
    else:
        array = np.asarray(value)
        if array.dtype.kind in {"U", "S", "O"}:
            data = np.array([_as_text(item) for item in array.ravel()], dtype=object).reshape(array.shape)
            dataset = parent.create_dataset(name, data=data, dtype=_STRING_DTYPE)
        else:
            dataset = parent.create_dataset(name, data=array)

    for key, item in attrs.items():
        dataset.attrs[key] = _as_text(item) if isinstance(item, (dict, list, tuple)) else item

    return dataset


def _write_table(parent: h5py.Group, name: str, rows: list[dict[str, Any]]) -> h5py.Group:
    group = _replace_group(parent, name)
    group.attrs["kind"] = "table"
    group.attrs["row_count"] = len(rows)

    if not rows:
        group.attrs["columns"] = []
        return group

    columns = _columns(rows)
    group.attrs["columns"] = columns

    for column in columns:
        values = [row.get(column) for row in rows]

        if _is_numeric_column(values):
            data = np.array([_as_float(value) for value in values], dtype=float)
            group.create_dataset(safe_group_name(column), data=data)
        else:
            data = np.array([_as_text(value) for value in values], dtype=object)
            group.create_dataset(safe_group_name(column), data=data, dtype=_STRING_DTYPE)

    return group


def _numeric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number


def solution_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return solution records with a numeric helper column for plotting."""
    normalized: list[dict[str, Any]] = []

    for record in records:
        row = {str(key): _plain(value) for key, value in record.items()}
        if "value" in row and "numeric_value" not in row:
            row["numeric_value"] = _numeric_value(row["value"])
        normalized.append(row)

    return normalized


def model_configuration(models: list[Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for model in models or []:
        rows.append(
            {
                "model": getattr(model, "name", "<unknown>"),
                "active_option": getattr(model, "active_option_name", None),
                "available_options": list(getattr(model, "available_options", [])),
                "order": list(getattr(model, "order", [])),
            }
        )

    return rows


def _records_without_tracks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in records if row.get("component_type") != "TrackedState"]


def _write_component_values(parent: h5py.Group, records: list[dict[str, Any]]) -> h5py.Group:
    components_group = _replace_group(parent, "components")
    components_group.attrs["kind"] = "components"

    for row in _records_without_tracks(solution_records(records)):
        component_name = row.get("component_name", "<unknown>")
        component_type = row.get("component_type", "<unknown>")
        attribute = row.get("attribute", "value")
        value = row.get("value")
        numeric_value = row.get("numeric_value")

        component_group = components_group.require_group(safe_group_name(component_name))
        component_group.attrs["name"] = str(component_name)
        component_group.attrs["type"] = str(component_type)

        if numeric_value is not None and math.isfinite(_as_float(numeric_value)):
            dataset_value = float(numeric_value)
        else:
            dataset_value = value

        _write_dataset(
            component_group,
            str(attribute),
            dataset_value,
            attrs={
                "attribute": str(attribute),
                "component_name": str(component_name),
                "component_type": str(component_type),
            },
        )

    return components_group


def _ordered_times(rows: list[dict[str, Any]]) -> np.ndarray:
    times: list[float] = []
    seen: set[float] = set()
    for row in rows:
        if "time" not in row:
            continue
        time_value = float(row["time"])
        if time_value in seen:
            continue
        seen.add(time_value)
        times.append(time_value)
    return np.asarray(times, dtype=float)


def _write_component_history(parent: h5py.Group, rows: list[dict[str, Any]], time_values: np.ndarray) -> h5py.Group:
    components_group = _replace_group(parent, "components")
    components_group.attrs["kind"] = "component_histories"

    time_index = {float(value): i for i, value in enumerate(time_values)}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for row in _records_without_tracks(solution_records(rows)):
        key = (
            str(row.get("component_name", "<unknown>")),
            str(row.get("component_type", "<unknown>")),
            str(row.get("attribute", "value")),
        )
        grouped.setdefault(key, []).append(row)

    for (component_name, component_type, attribute), items in grouped.items():
        numeric_values = [_as_float(row.get("numeric_value")) for row in items]
        numeric = any(math.isfinite(value) for value in numeric_values)

        component_group = components_group.require_group(safe_group_name(component_name))
        component_group.attrs["name"] = component_name
        component_group.attrs["type"] = component_type

        if numeric:
            data = np.full(len(time_values), np.nan, dtype=float)
            for row in items:
                i = time_index.get(float(row["time"]))
                if i is not None:
                    data[i] = _as_float(row.get("numeric_value"))
        else:
            data = np.full(len(time_values), "", dtype=object)
            for row in items:
                i = time_index.get(float(row["time"]))
                if i is not None:
                    data[i] = _as_text(row.get("value"))

        _write_dataset(
            component_group,
            attribute,
            data,
            attrs={
                "attribute": attribute,
                "component_name": component_name,
                "component_type": component_type,
            },
        )

    return components_group


def _write_tracks(parent: h5py.Group, rows: list[dict[str, Any]], time_values: np.ndarray) -> h5py.Group:
    tracks_group = _replace_group(parent, "tracks")
    tracks_group.attrs["kind"] = "tracked_histories"

    time_index = {float(value): i for i, value in enumerate(time_values)}
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in solution_records(rows):
        track_name = str(row.get("attribute", "track"))
        grouped.setdefault(track_name, []).append(row)

    for track_name, items in grouped.items():
        numeric_values = [_as_float(row.get("numeric_value")) for row in items]
        numeric = any(math.isfinite(value) for value in numeric_values)

        if numeric:
            data = np.full(len(time_values), np.nan, dtype=float)
            for row in items:
                i = time_index.get(float(row["time"]))
                if i is not None:
                    data[i] = _as_float(row.get("numeric_value"))
        else:
            data = np.full(len(time_values), "", dtype=object)
            for row in items:
                i = time_index.get(float(row["time"]))
                if i is not None:
                    data[i] = _as_text(row.get("value"))

        _write_dataset(tracks_group, track_name, data, attrs={"name": track_name})

    return tracks_group


def _section_from_group_path(group_path: str) -> str:
    text = str(group_path).strip("/").lower()
    if "transient" in text and "final" in text:
        return "transient/final"
    if text.endswith("final"):
        return "transient/final"
    if "model" in text:
        return text
    return "steady_state"


def write_solution(
    target: str | Path | HDF5Target,
    records: list[dict[str, Any]],
    *,
    network_name: str | None = None,
    models: list[Any] | None = None,
    group_path: str = "solution/current",
) -> Path:
    """Write one exported network solution to HDF5.

    The default writes current steady-state data to::

        /<network_name>/steady_state

    Passing ``group_path`` containing ``"final"`` writes the records to::

        /<network_name>/transient/final
    """
    if isinstance(target, HDF5Target):
        path = hdf5_path(target.filename)
        section = target.group_path.strip("/") or _section_from_group_path(group_path)
    else:
        path = hdf5_path(target)
        section = _section_from_group_path(group_path)

    if network_name is None:
        network_name = "network"

    rows = solution_records(records)

    with h5py.File(path, "a") as h5:
        _initialize_file(h5)
        network_group = _require_object_group(h5, network_name, "network")

        section_group = network_group.require_group(section)
        section_group.attrs["kind"] = section.rsplit("/", 1)[-1]
        section_group.attrs["updated_utc"] = _now()

        _delete_children(section_group, ["components", "table", "model_configuration"])
        _write_component_values(section_group, rows)
        _write_table(section_group, "table", rows)

        model_rows = model_configuration(models)
        if model_rows:
            _write_table(section_group, "model_configuration", model_rows)

    return path


def write_transient_solution(
    filename: str | Path,
    *,
    network_name: str,
    history_records: list[dict[str, Any]],
    track_records: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    models: list[Any] | None = None,
    output_times: list[float] | None = None,
) -> Path:
    """Write current transient data for a network.

    This overwrites only ``/<network_name>/transient`` and keeps other top-level
    objects, maps, and other networks in the same file untouched.
    """
    path = hdf5_path(filename)
    history_rows = solution_records(history_records)
    track_rows = solution_records(track_records)
    final_rows = solution_records(final_records)
    if output_times is None:
        time_values = _ordered_times(history_rows + track_rows)
    else:
        time_values = np.asarray(output_times, dtype=float)

    with h5py.File(path, "a") as h5:
        _initialize_file(h5)
        network_group = _require_object_group(h5, network_name, "network")
        if "transient" in network_group:
            del network_group["transient"]
        transient_group = network_group.create_group("transient")
        transient_group.attrs["kind"] = "transient"
        transient_group.attrs["updated_utc"] = _now()

        _write_dataset(transient_group, "time", time_values, attrs={"name": "time"})
        _write_component_history(transient_group, history_rows, time_values)
        _write_tracks(transient_group, track_rows, time_values)
        _write_table(transient_group, "table", history_rows)
        _write_table(transient_group, "diagnostics", step_rows)

        final_group = transient_group.create_group("final")
        final_group.attrs["kind"] = "final"
        _write_component_values(final_group, final_rows)
        _write_table(final_group, "table", final_rows)

        model_rows = model_configuration(models)
        if model_rows:
            _write_table(final_group, "model_configuration", model_rows)

    return path


def write_tables(
    target: str | Path | HDF5Target,
    tables: dict[str, list[dict[str, Any]]],
    *,
    group_path: str = "statistics/current",
) -> Path:
    """Write a dictionary of row tables to HDF5.

    When ``target`` is an :class:`HDF5Target`, its group path is used exactly.
    Otherwise ``group_path`` is used exactly at the file root. Solver code passes
    explicit object-scoped targets for user-facing exports.
    """
    if isinstance(target, HDF5Target):
        path = hdf5_path(target.filename)
        group_path = target.group_path.strip("/")
    else:
        path = hdf5_path(target)
        group_path = group_path.strip("/")

    with h5py.File(path, "a") as h5:
        _initialize_file(h5)
        parent = h5.require_group(group_path)
        parent.attrs["updated_utc"] = _now()

        for name, rows in tables.items():
            _write_table(parent, name, rows)

    return path


def write_model_option_results(
    filename: str | Path,
    results: dict[str, list[dict[str, Any]]],
    *,
    model_name: str,
    network_name: str | None = None,
) -> Path:
    """Write every successful model-option solution into one HDF5 file."""
    path = hdf5_path(filename)
    network_name = network_name or "network"

    with h5py.File(path, "a") as h5:
        _initialize_file(h5)
        network_group = _require_object_group(h5, network_name, "network")
        model_group = network_group.require_group("model_options").require_group(safe_group_name(model_name))
        model_group.attrs["name"] = model_name

        for option_name, records in results.items():
            option_group = _replace_group(model_group, option_name)
            option_group.attrs["name"] = str(option_name)
            rows = solution_records(records)
            _write_component_values(option_group, rows)
            _write_table(option_group, "table", rows)

    return path


def write_failures(
    filename: str | Path,
    failures: list[Any],
    *,
    group_path: str = "failures",
) -> Path:
    path = hdf5_path(filename)
    rows = [
        {
            "model": getattr(failure, "model", ""),
            "option": getattr(failure, "option", ""),
            "error_type": getattr(failure, "error_type", ""),
            "error": getattr(failure, "error", ""),
        }
        for failure in failures
    ]
    write_tables(path, {"table": rows}, group_path=group_path)
    return path
