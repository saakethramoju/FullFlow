"""Transient result collection and HDF5 export helpers."""

from __future__ import annotations

from typing import Any

from fullflow.Exports.HDF5 import write_transient_solution


def format_records(records: list[dict[str, Any]], return_type: str):
    """Return transient records in the requested public format."""
    return_type = return_type.lower()

    if return_type in {"dict", "records"}:
        return records

    raise ValueError("return_type must be 'dict'.")


class TransientHistory:
    """Collect full-network transient history at each accepted timestep.

    The HDF5 export writes every component and balance attribute at every saved
    time.  ``network.track(...)`` is still supported, but it only creates
    convenient aliases under ``/<network>/transient/tracks``; it is no longer
    required for plotting or reading transient component histories.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.track_records: list[dict[str, Any]] = []

    def append(self, network, time_value: float) -> None:
        """Append current full-network and tracked values at ``time_value``."""
        for record in network.save(filename=None, return_type="dict"):
            row = {"time": float(time_value)}
            row.update(record)
            self.records.append(row)

        for record in network.tracked_records():
            row = {"time": float(time_value)}
            row.update(record)
            self.track_records.append(row)

    def save(self, filename: str | None, network, step_rows: list[dict[str, Any]]) -> None:
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
        )
