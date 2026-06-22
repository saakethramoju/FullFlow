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
    """Collect user-tracked values at each accepted timestep.

    Transient history intentionally stores only values registered with
    ``network.track(...)``.  The final full-network state is still written under
    the transient solution's ``/final`` subgroup.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, network, time_value: float) -> None:
        """Append the current tracked values at ``time_value``."""
        for record in network.tracked_records():
            row = {"time": float(time_value)}
            row.update(record)
            self.records.append(row)

    def save(
        self,
        filename: str | None,
        network,
        step_rows: list[dict[str, Any]],
        *,
        group_path: str = "auto",
        solution_name: str | None = None,
    ) -> None:
        """Write transient history, step diagnostics, and final state to HDF5.

        New layout
        ----------
        ``/solutions/transient_####/history``
            Time-stamped tracked records in long-table form.

        ``/solutions/transient_####/tracks``
            One numeric dataset per tracked variable, plus a shared ``time``
            dataset. This is the convenient plotting view.

        ``/solutions/transient_####/steps``
            One row per accepted timestep containing residual norms, solve
            timing, and SciPy iteration counts.

        ``/solutions/transient_####/final/records``
            Final full-network state in the same table format as steady-state
            saves.
        """
        if filename is None:
            return

        write_transient_solution(
            filename,
            self.records,
            step_rows,
            network.save(filename=None, return_type="dict"),
            network_name=network.name,
            models=network.model_list,
            group_path=group_path,
            name=solution_name,
        )
