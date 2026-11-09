"""Operating-point records and consistent Python result dictionaries."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np


class SolutionRecords(list):
    """Raw exported rows with optional restorable operating points."""

    def __init__(self, records=()):
        super().__init__()
        self.operating_points = {}
        self.add_records(records)

    def add_records(self, records, time_value=None):
        from .linearization.checkpoints import _copy

        rows = []
        for record in records:
            row = dict(record)
            row["value"] = _copy(record["value"])
            if time_value is not None:
                row["time"] = float(time_value)
            rows.append(row)
        self.extend(rows)
        return rows

    def capture(self, network, cache, issue=None):
        from .linearization.checkpoints import OperatingCheckpoint

        point = OperatingCheckpoint(network, cache)
        self.operating_points[point.time] = (point, issue)
        return point


def _snapshot(value: Any) -> Any:
    from .linearization.checkpoints import _copy
    return _copy(value)


def _scalar(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (Real, np.number)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if np.isfinite(number):
            return number
    return None


def _initial(value: Any, time_value: float | None) -> dict[str, Any]:
    number = _scalar(value)
    entry = {"initial": _snapshot(value), "final": _snapshot(value)}
    if number is not None:
        entry.update(min=number, max=number, time_min=time_value, time_max=time_value)
    return entry


def update_entry(entry: dict[str, Any], value: Any, time_value: float | None) -> None:
    """Update scalar extrema only at accepted operating points."""
    entry["final"] = _snapshot(value)
    number = _scalar(value)
    if number is None:
        return
    if "min" not in entry or number < entry["min"]:
        entry["min"] = number
        entry["time_min"] = time_value
    if "max" not in entry or number > entry["max"]:
        entry["max"] = number
        entry["time_max"] = time_value


def organized_records(records, *, full: bool = False, steady: bool = False) -> dict[str, Any]:
    """Organize existing export records by tracked name and Component attribute."""
    grouped: dict[str, dict[str, Any]] = {
        "tracked": {}, "components": {}, "balances": {},
    }
    series: dict[tuple[str, str, str], tuple[list[float], list[Any]]] = {}
    for row in records:
        owner = row["component_name"]
        attr = row["attribute"]
        kind = row["component_type"]
        bucket = "tracked" if kind == "TrackedState" else "balances" if kind == "Balance" else "components"
        parent = grouped[bucket] if bucket == "tracked" else grouped[bucket].setdefault(owner, {})
        key = attr
        time_value = row.get("time")
        value = row["value"]
        if steady:
            parent[key] = {"final": _snapshot(value)}
            continue
        if key not in parent:
            parent[key] = _initial(value, time_value)
            if full:
                parent[key]["initial"] = value
                parent[key]["final"] = value
        else:
            if full:
                entry = parent[key]
                entry["final"] = value
                number = _scalar(value)
                if number is not None:
                    if "min" not in entry or number < entry["min"]:
                        entry["min"], entry["time_min"] = number, time_value
                    if "max" not in entry or number > entry["max"]:
                        entry["max"], entry["time_max"] = number, time_value
            else:
                update_entry(parent[key], value, time_value)
        if full and time_value is not None:
            times, values = series.setdefault((bucket, owner, key), ([], []))
            times.append(float(time_value))
            values.append(value)

    if full:
        for (bucket, owner, key), (times, values) in series.items():
            entry = grouped[bucket][key] if bucket == "tracked" else grouped[bucket][owner][key]
            entry["time"] = np.asarray(times, dtype=float)
            if all(_scalar(value) is not None for value in values):
                entry["values"] = np.asarray(values, dtype=float)
            else:
                entry["values"] = values

    return grouped


def steady_result(records, *, network, diagnostics=None, metadata=None, elapsed=None):
    data = organized_records(records, steady=True)
    sol = getattr(diagnostics, "sol", None)
    raw_residual = None if sol is None else getattr(sol, "fun", None)
    residual_values = None if raw_residual is None else np.asarray(raw_residual, dtype=float)
    data.update(
        time=float(network.time.value),
        history="steady",
        events={"sensor": network.sensor_event_records(), "sequence": []},
        solver={
            "success": True,
            "final_time": float(network.time.value),
            "elapsed_s": float(elapsed if elapsed is not None else getattr(diagnostics, "elapsed_time", 0.0)),
            "status": None if sol is None else getattr(sol, "status", None),
            "message": "Static evaluation" if sol is None else str(getattr(sol, "message", "")),
            "function_evaluations": None if sol is None else getattr(sol, "nfev", None),
            "jacobian_evaluations": None if sol is None else getattr(sol, "njev", None),
            "max_residual": None if residual_values is None else (
                float(np.max(np.abs(residual_values))) if residual_values.size else 0.0
            ),
            "cost": None if sol is None else getattr(sol, "cost", None),
            "optimality": None if sol is None else getattr(sol, "optimality", None),
            "variable_count": None if sol is None else len(getattr(sol, "x", ())),
            "residual_count": None if residual_values is None else residual_values.size,
        },
    )
    return data


class SummaryAccumulator:
    """Streaming tracked-value summaries with constant per-track storage."""

    def __init__(self):
        self.tracked: dict[str, dict[str, Any]] = {}
        self.samples = 0

    def add(self, records, time_value: float) -> None:
        for record in records:
            attr = record["attribute"]
            value = record["value"]
            if attr not in self.tracked:
                self.tracked[attr] = _initial(value, time_value)
            else:
                update_entry(self.tracked[attr], value, time_value)
        self.samples += 1
