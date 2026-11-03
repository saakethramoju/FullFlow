"""Transient result collection and HDF5 export helpers."""

from __future__ import annotations

from typing import Any

from fullflow.Exports.HDF5 import write_transient_solution
from fullflow.Solvers.results import SolutionRecords


def format_records(records: list[dict[str, Any]], return_type: str):
    """Return transient records in the requested public format."""
    return_type = return_type.lower()

    if return_type in {"dict", "records"}:
        return records

    raise ValueError("return_type must be 'dict' or 'records'.")


class TransientHistory:
    """Collect output and operating points at accepted timesteps."""

    def __init__(self) -> None:
        """Initialize the object and register any FullFlow state wiring."""
        self.records = SolutionRecords()
        self.track_records: list[dict[str, Any]] = []
        self.output_times: list[float] = []

    @property
    def public_records(self) -> list[dict[str, Any]]:
        """Return the records that should be returned from ``Transient.solve``."""
        return self.records

    def append(self, network, time_value: float, cache, issue=None) -> None:
        """Record the accepted model state and its exported values."""
        time_value = float(time_value)
        self.records.capture(network, cache, issue)
        self.output_times.append(time_value)

        rows = self.records.add_records(
            network.save(filename=None, return_type="dict"), time_value,
        )
        self.track_records.extend(row for row in rows if row["component_type"] == "TrackedState")

    def save(
        self,
        filename: str | None,
        network,
        step_rows: list[dict[str, Any]],
        *,
        group_path: str = "transient/runs/base",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write transient history, diagnostics, and final state to HDF5."""
        if filename is None:
            return

        write_transient_solution(
            filename,
            network_name=network.name,
            history_records=self.records,
            track_records=self.track_records,
            step_rows=step_rows,
            final_records=network.save(filename=None, return_type="dict"),
            models=network.model_list,
            output_times=self.output_times,
            sensor_event_rows=network.sensor_event_records(),
            sensor_condition_trace_rows=network.sensor_condition_trace_records(),
            sequence_command_trace_rows=network.sequence_command_trace_records(),
            group_path=group_path,
            metadata=metadata,
        )
