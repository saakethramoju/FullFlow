"""HDF5 export helpers for FullFlow.

FullFlow stores exported network solutions and solver statistics in one HDF5
file. The public API accepts a base filename; ``.h5`` is added automatically
when no HDF5 extension is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any
import json
import math
import re

import h5py
import numpy as np


HDF5_EXTENSIONS = {".h5", ".hdf5"}
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


def _write_table(parent: h5py.Group, name: str, rows: list[dict[str, Any]]) -> None:
    group = _replace_group(parent, name)
    group.attrs["row_count"] = len(rows)

    if not rows:
        group.attrs["columns"] = []
        return

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


def write_solution(
    target: str | Path | HDF5Target,
    records: list[dict[str, Any]],
    *,
    network_name: str | None = None,
    models: list[Any] | None = None,
    group_path: str = "solution/current",
) -> Path:
    """Write one exported network solution table to HDF5."""
    path, group_path = _target_path_and_group(target, group_path)

    with h5py.File(path, "a") as h5:
        h5.attrs["fullflow_export_format"] = "fullflow-hdf5-v1"
        if network_name is not None:
            h5.attrs["network_name"] = network_name

        group = h5.require_group(group_path)
        _write_table(group, "records", solution_records(records))

        model_rows = model_configuration(models)
        if model_rows:
            _write_table(group, "model_configuration", model_rows)

    return path


def write_tables(
    target: str | Path | HDF5Target,
    tables: dict[str, list[dict[str, Any]]],
    *,
    group_path: str = "statistics/current",
) -> Path:
    """Write a dictionary of row tables to HDF5."""
    path, group_path = _target_path_and_group(target, group_path)

    with h5py.File(path, "a") as h5:
        h5.attrs["fullflow_export_format"] = "fullflow-hdf5-v1"
        parent = h5.require_group(group_path)

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
    safe_model = safe_group_name(model_name)

    for option_name, records in results.items():
        target = HDF5Target(path, f"models/{safe_model}/{safe_group_name(option_name)}/solution")
        write_solution(target, records, network_name=network_name)

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
    write_tables(path, {"records": rows}, group_path=group_path)
    return path
