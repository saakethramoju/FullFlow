"""HDF5 export helpers for FullFlow.

The FullFlow HDF5 layout is intentionally simple and consistent across
steady-state solutions, transient solutions, generated maps, solver diagnostics,
and model-option sweeps.

Recommended layout
------------------

Root
    attrs
        fullflow_export_format = "fullflow-hdf5-v2"
        fullflow_schema_version = 2

/solutions
    /steady_state_0001
        attrs["fullflow_kind"] = "steady_state_solution"
        /records
        /model_configuration

    /transient_0001
        attrs["fullflow_kind"] = "transient_solution"
        /history
        /tracks
        /steps
        /final/records
        /final/model_configuration

    /model_options/<model>/<option>
        attrs["fullflow_kind"] = "steady_state_solution"
        /records

/maps
    /<map_name>
        attrs["fullflow_kind"] = "map"
        /axes
        /outputs
        /status

/diagnostics
    Solver diagnostics and statistics.

Each table group stores one dataset per column and includes ``columns`` and
``row_count`` attributes. Numeric columns are stored as floating-point arrays;
non-numeric columns are stored as UTF-8 strings. This makes the file easy to
inspect with HDFView, h5py, MATLAB, or simple plotting utilities.
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
HDF5_FORMAT = "fullflow-hdf5-v2"
HDF5_SCHEMA_VERSION = 2
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


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


def solution_group_name(kind: str, name: str | None = None) -> str:
    """Return a user-readable solution group name prefix."""
    if name is not None and str(name).strip():
        return safe_group_name(name)

    kind = str(kind).lower().strip()

    if "transient" in kind:
        return "transient"

    if "steady" in kind:
        return "steady_state"

    return safe_group_name(kind or "solution")


def map_group_path(group: str) -> str:
    """Return the canonical map group path for a user-supplied map name."""
    group = str(group).strip("/") or "map"

    if group == "maps" or group.startswith("maps/"):
        return group

    return f"maps/{safe_group_name(group)}"


def _set_file_attrs(h5: h5py.File, *, network_name: str | None = None) -> None:
    h5.attrs["fullflow_export_format"] = HDF5_FORMAT
    h5.attrs["fullflow_schema_version"] = HDF5_SCHEMA_VERSION

    if "created_utc" not in h5.attrs:
        h5.attrs["created_utc"] = _datetime.datetime.now(_datetime.UTC).isoformat()

    h5.attrs["updated_utc"] = _datetime.datetime.now(_datetime.UTC).isoformat()

    if network_name is not None:
        h5.attrs["network_name"] = network_name


def _target_path_and_group(target: str | Path | HDF5Target, default_group: str) -> tuple[Path, str]:
    if isinstance(target, HDF5Target):
        return hdf5_path(target.filename), target.group_path.strip("/") or default_group
    return hdf5_path(target), default_group.strip("/")


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
        if value is None:
            continue

        if isinstance(value, bool):
            continue

        if isinstance(value, np.bool_):
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




def _replace_link(h5: h5py.File, link_path: str, target) -> None:
    """Replace ``link_path`` with a soft link to ``target``.

    This keeps old user scripts working while the canonical data lives in the
    v2 layout. For example, ``/transient`` can point to the newest
    ``/solutions/transient_####`` group without confusing canonical group
    discovery.
    """
    link_path = str(link_path).strip("/")
    if not link_path:
        return

    parent_path, _, name = link_path.rpartition("/")
    parent = h5.require_group(parent_path) if parent_path else h5

    if name in parent:
        del parent[name]

    parent[name] = h5py.SoftLink("/" + target.name.strip("/"))


def _write_json_attrs(group: h5py.Group, metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return

    for key, value in metadata.items():
        key = str(key)

        if isinstance(value, (str, int, float, bool, np.number, np.bool_)):
            group.attrs[key] = _plain(value)
        else:
            group.attrs[key] = json.dumps(_plain(value), ensure_ascii=False, default=str)


def _write_table(parent: h5py.Group, name: str, rows: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> h5py.Group:
    group = _replace_group(parent, name)
    group.attrs["fullflow_kind"] = "table"
    group.attrs["row_count"] = len(rows)
    _write_json_attrs(group, metadata)

    if not rows:
        group.attrs["columns"] = []
        return group

    columns = _columns(rows)
    group.attrs["columns"] = columns

    for column in columns:
        values = [row.get(column) for row in rows]
        dataset_name = safe_group_name(column)

        if _is_numeric_column(values):
            data = np.array([_as_float(value) for value in values], dtype=float)
            dataset = group.create_dataset(dataset_name, data=data)
            dataset.attrs["column"] = column
        else:
            data = np.array([_as_text(value) for value in values], dtype=object)
            dataset = group.create_dataset(dataset_name, data=data, dtype=_STRING_DTYPE)
            dataset.attrs["column"] = column

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


def _next_group_name(parent: h5py.Group, prefix: str) -> str:
    prefix = safe_group_name(prefix)
    index = 1

    while True:
        name = f"{prefix}_{index:04d}"
        if name not in parent:
            return name
        index += 1


def _resolve_auto_group(h5: h5py.File, collection: str, prefix: str, name: str | None = None) -> str:
    parent = h5.require_group(collection.strip("/"))

    if name is not None and str(name).strip():
        base_name = safe_group_name(name)
        if base_name not in parent:
            return f"{collection.strip('/')}/{base_name}"

        index = 2
        while f"{base_name}_{index:04d}" in parent:
            index += 1
        return f"{collection.strip('/')}/{base_name}_{index:04d}"

    return f"{collection.strip('/')}/{_next_group_name(parent, prefix)}"


def write_solution(
    target: str | Path | HDF5Target,
    records: list[dict[str, Any]],
    *,
    network_name: str | None = None,
    models: list[Any] | None = None,
    group_path: str = "auto",
    kind: str = "steady_state",
    name: str | None = None,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one exported network solution table to HDF5.

    When ``group_path='auto'``, a new numbered group is created under
    ``/solutions``.  This avoids overwriting old steady-state and transient
    cases when the same file is reused.
    """
    path, requested_group = _target_path_and_group(target, group_path)

    with h5py.File(path, "a") as h5:
        _set_file_attrs(h5, network_name=network_name)

        if requested_group == "auto":
            requested_group = _resolve_auto_group(
                h5,
                "solutions",
                solution_group_name(kind, None),
                name=name,
            )

        if overwrite and requested_group in h5:
            del h5[requested_group]

        group = h5.require_group(requested_group)
        group.attrs["fullflow_kind"] = f"{kind}_solution" if not str(kind).endswith("solution") else kind
        group.attrs["kind"] = kind
        group.attrs["created_utc"] = _datetime.datetime.now(_datetime.UTC).isoformat()
        if network_name is not None:
            group.attrs["network_name"] = network_name
        _write_json_attrs(group, metadata)

        _write_table(group, "records", solution_records(records))

        model_rows = model_configuration(models)
        if model_rows:
            _write_table(group, "model_configuration", model_rows)

        if str(kind).startswith("steady_state"):
            _replace_link(h5, "solution/final", group)

    return path


def write_tables(
    target: str | Path | HDF5Target,
    tables: dict[str, list[dict[str, Any]]],
    *,
    group_path: str = "diagnostics/current",
    kind: str = "tables",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a dictionary of row tables to HDF5."""
    path, group_path = _target_path_and_group(target, group_path)

    with h5py.File(path, "a") as h5:
        _set_file_attrs(h5)
        parent = h5.require_group(group_path)
        parent.attrs["fullflow_kind"] = kind
        _write_json_attrs(parent, metadata)

        for name, rows in tables.items():
            _write_table(parent, name, rows)

    return path


def _write_track_datasets(solution_group: h5py.Group, history_rows: list[dict[str, Any]]) -> None:
    tracks_group = _replace_group(solution_group, "tracks")
    tracks_group.attrs["fullflow_kind"] = "transient_tracks"

    time_values: dict[float, None] = {}
    track_rows: dict[str, list[dict[str, Any]]] = {}

    for row in solution_records(history_rows):
        if "time" not in row or "attribute" not in row:
            continue

        time = _as_float(row.get("time"))
        if not math.isfinite(time):
            continue

        attribute = _as_text(row.get("attribute"))
        if not attribute:
            continue

        time_values[time] = None
        track_rows.setdefault(attribute, []).append(row)

    time = np.array(sorted(time_values), dtype=float)
    time_dataset = tracks_group.create_dataset("time", data=time)
    time_dataset.attrs["long_name"] = "Time"
    time_dataset.attrs["axis"] = "/" + time_dataset.name

    used_names: set[str] = {"time"}

    for attribute, rows in track_rows.items():
        values_by_time = {
            _as_float(row.get("time")): _as_float(row.get("numeric_value"))
            for row in rows
        }

        values = np.array([values_by_time.get(item, math.nan) for item in time], dtype=float)
        dataset_name = safe_group_name(attribute)
        original_name = dataset_name
        suffix = 2

        while dataset_name in used_names:
            dataset_name = f"{original_name}_{suffix}"
            suffix += 1

        used_names.add(dataset_name)
        dataset = tracks_group.create_dataset(dataset_name, data=values)
        dataset.attrs["long_name"] = attribute
        dataset.attrs["label"] = attribute
        dataset.attrs["axis"] = "/" + time_dataset.name


def write_transient_solution(
    filename: str | Path,
    history_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    *,
    network_name: str | None = None,
    models: list[Any] | None = None,
    group_path: str = "auto",
    name: str | None = None,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one complete transient solution to a new solution group."""
    path = hdf5_path(filename)

    with h5py.File(path, "a") as h5:
        _set_file_attrs(h5, network_name=network_name)

        if group_path == "auto":
            group_path = _resolve_auto_group(h5, "solutions", "transient", name=name)
        else:
            group_path = group_path.strip("/")

        if overwrite and group_path in h5:
            del h5[group_path]

        solution_group = h5.require_group(group_path)
        solution_group.attrs["fullflow_kind"] = "transient_solution"
        solution_group.attrs["kind"] = "transient"
        solution_group.attrs["created_utc"] = _datetime.datetime.now(_datetime.UTC).isoformat()
        if network_name is not None:
            solution_group.attrs["network_name"] = network_name
        _write_json_attrs(solution_group, metadata)

        _write_table(solution_group, "history", solution_records(history_rows))
        _write_table(solution_group, "steps", step_rows)
        _write_track_datasets(solution_group, history_rows)

        final_group = _replace_group(solution_group, "final")
        final_group.attrs["fullflow_kind"] = "final_state"
        _write_table(final_group, "records", solution_records(final_records))

        model_rows = model_configuration(models)
        if model_rows:
            _write_table(final_group, "model_configuration", model_rows)

        _replace_link(h5, "transient", solution_group)
        _replace_link(h5, "solution/final", final_group)

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
    safe_model = safe_group_name(model_name)

    for option_name, records in results.items():
        option_group = f"solutions/model_options/{safe_model}/{safe_group_name(option_name)}"
        write_solution(
            HDF5Target(path, option_group),
            records,
            network_name=network_name,
            kind="steady_state_model_option",
            group_path=option_group,
        )

    return path


def write_failures(
    filename: str | Path,
    failures: list[Any],
    *,
    group_path: str = "diagnostics/failures",
) -> Path:
    rows = [
        {
            "model": getattr(failure, "model", ""),
            "option": getattr(failure, "option", ""),
            "error_type": getattr(failure, "error_type", ""),
            "error": getattr(failure, "error", ""),
        }
        for failure in failures
    ]
    return write_tables(filename, {"records": rows}, group_path=group_path, kind="failures")
