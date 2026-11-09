"""Full and lightweight transient result collection."""

from __future__ import annotations

import numpy as np

from fullflow.Exports.HDF5 import write_summary_solution, write_transient_solution
from fullflow.Solvers.results import SolutionRecords, SummaryAccumulator, organized_records


class TransientHistory:
    """Collect accepted operating points or streaming tracked summaries."""

    def __init__(self, mode: str = "full") -> None:
        if mode not in {"full", "summary"}:
            raise ValueError("history must be 'full' or 'summary'.")
        self.mode = mode
        self.records = SolutionRecords()
        self.summary = SummaryAccumulator()
        self.track_records: list[dict[str, Any]] = []
        self.output_times: list[float] = []

    def append(self, network, time_value: float, cache, issue=None) -> None:
        time_value = float(time_value)
        if self.mode == "summary":
            self.summary.add(network.tracked_records(), time_value)
            return
        self.records.capture(network, cache, issue)
        self.output_times.append(time_value)
        rows = self.records.add_records(network._save_records(), time_value)
        self.track_records.extend(row for row in rows if row["component_type"] == "TrackedState")

    def capture_final(self, network, cache, issue=None):
        """Retain one final checkpoint for summary-mode linearization."""
        if self.mode == "summary":
            return self.records.capture(network, cache, issue)
        return self.records.operating_points[float(network.time.value)][0]

    def result(self, network, *, metadata, solver, events):
        if self.mode == "full":
            data = organized_records(self.records, full=True)
            data["time"] = np.asarray(self.output_times, dtype=float)
        else:
            final = organized_records(network._save_records())
            data = {
                "tracked": self.summary.tracked,
                "components": final["components"],
                "balances": final["balances"],
            }
            # Newly tracked values are also present if the last step added tracks.
            for name, value in final["tracked"].items():
                data["tracked"].setdefault(name, value)

        data.update(
            history=self.mode,
            model={
                "name": metadata.get("model_name") or None,
                "selected_option": metadata.get("option_name") or None,
            },
            solver=solver,
            events=events,
            failures=[],
        )
        return data

    def save(
        self,
        filename: str | None,
        network,
        step_rows: list[dict[str, Any]],
        *,
        group_path: str = "transient/runs/base",
        metadata: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if filename is None:
            return

        if self.mode == "summary":
            if result is None:
                raise ValueError("Summary HDF5 export requires result data.")
            write_summary_solution(
                filename,
                network_name=network.name,
                result=result,
                final_records=network._save_records(),
                group_path=group_path,
                metadata=metadata,
                models=network.model_list,
            )
            return

        write_transient_solution(
            filename,
            network_name=network.name,
            history_records=self.records,
            track_records=self.track_records,
            step_rows=step_rows,
            final_records=network._save_records(),
            models=network.model_list,
            output_times=self.output_times,
            sensor_event_rows=network.sensor_event_records(),
            sensor_condition_trace_rows=network.sensor_condition_trace_records(),
            sequence_command_trace_rows=network.sequence_command_trace_records(),
            group_path=group_path,
            metadata=metadata,
        )


class DiagnosticSummary:
    """Constant-size counters for accepted timesteps."""

    def __init__(self):
        self.steps = 0
        self.retries = 0
        self.min_dt = None
        self.max_dt = None
        self.function_evaluations = 0
        self.jacobian_evaluations = 0
        self.cost = 0.0
        self.worst_residual = 0.0
        self.last = None

    def append(self, row):
        self.steps += 1
        self.retries += int(row.retries)
        self.min_dt = row.dt if self.min_dt is None else min(self.min_dt, row.dt)
        self.max_dt = row.dt if self.max_dt is None else max(self.max_dt, row.dt)
        self.worst_residual = max(self.worst_residual, row.max_residual)
        sol = row.sol
        if sol is not None:
            self.function_evaluations += int(getattr(sol, "nfev", 0) or 0)
            self.jacobian_evaluations += int(getattr(sol, "njev", 0) or 0)
            self.cost += float(getattr(sol, "cost", 0.0) or 0.0)
        self.last = row

    def result(self, *, final_time, elapsed, start_time, work, stop_reason):
        return {
            "success": stop_reason is None,
            "start_time": start_time,
            "final_time": final_time,
            "elapsed_s": elapsed,
            "accepted_steps": self.steps,
            "retries": self.retries,
            "minimum_dt": self.min_dt,
            "maximum_dt": self.max_dt,
            "function_evaluations": self.function_evaluations,
            "jacobian_evaluations": self.jacobian_evaluations,
            "max_residual": self.worst_residual,
            "final_residual": None if self.last is None else self.last.max_residual,
            "stop_reason": stop_reason,
            "work": dict(work),
        }
