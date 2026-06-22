"""Transient result collection and HDF5 export helpers."""

from __future__ import annotations

from typing import Any

from fullflow.Exports.HDF5 import write_solution, write_tables


def format_records(records: list[dict[str, Any]], return_type: str):
    """Return transient records in the requested public format."""
    return_type = return_type.lower()

    if return_type in {"dict", "records"}:
        return records

    raise ValueError("return_type must be 'dict'.")


class TransientHistory:
    """Collect one exported network snapshot per accepted timestep.

    For now, history uses the existing ``Network.save`` record format and adds a
    ``time`` column to each row.  That gives simple, general HDF5 output without
    creating a separate transient tracking API yet.  Users can still keep output
    compact by using ``network.track(...)`` and ignoring component attributes in
    later plotting scripts.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, network, time_value: float) -> None:
        """Append the current network state at ``time_value``."""
        for record in network.save(return_type="dict"):
            row = {"time": float(time_value)}
            row.update(record)
            self.records.append(row)

    def save(self, filename: str | None, network, step_rows: list[dict[str, Any]]) -> None:
        """Write transient history, step diagnostics, and final state to HDF5.

        HDF5 layout
        -----------
        ``/transient/history``
            Time-stamped network records for every accepted timestep, including
            the initial condition.

        ``/transient/steps``
            One row per accepted timestep containing ``time``, ``dt``, residual
            norms, solve time, and SciPy iteration counts.

        ``/solution/final``
            Final network state in the same table format used by steady-state
            saves.  This makes it easy to inspect the final state without
            filtering the full transient history.
        """
        if filename is None:
            return

        write_tables(
            filename,
            {
                "history": self.records,
                "steps": step_rows,
            },
            group_path="transient",
        )
        write_solution(
            filename,
            network.save(return_type="dict"),
            network_name=network.name,
            models=network.model_list,
            group_path="solution/final",
        )
