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
    """Collect full transient output at selected accepted timesteps.

    ``save_dt`` controls which accepted times are stored.  Whenever a time is
    saved, the full network export is stored.  ``network.track(...)`` values are
    also stored in the convenience ``/tracks`` HDF5 group, but tracking does not
    decide which states are saved.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.track_records: list[dict[str, Any]] = []
        self.output_times: list[float] = []

    @property
    def public_records(self) -> list[dict[str, Any]]:
        """Return the records that should be returned from ``Transient.solve``."""
        return self.records

    def append(self, network, time_value: float) -> None:
        """Append current full-network output values at ``time_value``."""
        time_value = float(time_value)
        self.output_times.append(time_value)

        for record in network.save(filename=None, return_type="dict"):
            row = {"time": time_value}
            row.update(record)
            self.records.append(row)

        for record in network.tracked_records():
            row = {"time": time_value}
            row.update(record)
            self.track_records.append(row)

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
            group_path=group_path,
            metadata=metadata,
        )
