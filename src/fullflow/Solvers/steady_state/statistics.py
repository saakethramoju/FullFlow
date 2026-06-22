"""Per-evaluation steady-state solver statistics.

This module records compact terminal progress during nonlinear solves and writes
complete plot-friendly statistics files when requested. It stays solver-side so
networks and components do not need to know anything about logging or exports.
"""

from __future__ import annotations

from typing import Any
import math
import time

import numpy as np
from rich.console import Console
from rich import box
from rich.table import Table
from rich.text import Text

from fullflow.Exports.HDF5 import HDF5Target, safe_group_name, write_tables


def _plain(value: Any) -> Text:
    """Return literal Rich text so user labels are never parsed as markup."""
    return Text(str(value))


class SolverStatistics:
    """Collect detailed per-evaluation nonlinear-solver statistics."""

    def __init__(self, enabled: bool = False, console: Console | None = None) -> None:
        self.enabled = enabled
        self.console = console or Console()
        self.reset()

    def reset(self) -> None:
        self.start_time = time.perf_counter()
        self.evaluation = 0
        self.previous_time: float | None = None
        self.previous_x: np.ndarray | None = None
        self.previous_cost: float | None = None
        self.initial_x: np.ndarray | None = None
        self.evaluations: list[dict[str, Any]] = []
        self.evaluation_metrics: list[dict[str, Any]] = []
        self.residuals: list[dict[str, Any]] = []
        self.variables: list[dict[str, Any]] = []
        self.network_records: list[dict[str, Any]] = []
        self.variable_metadata: list[dict[str, Any]] = []
        self.residual_metadata: list[dict[str, Any]] = []
        self.settings: list[dict[str, Any]] = []
        self.final_summary: list[dict[str, Any]] = []

    def configure(
        self,
        cache,
        least_squares_settings=None,
        state_settings=None,
        x0: Any | None = None,
    ) -> None:
        if not self.enabled:
            return

        if x0 is not None:
            self.initial_x = np.array(x0, dtype=float)

        if least_squares_settings is not None:
            self.settings.extend(
                [
                    {"setting": "solver_method", "value": least_squares_settings.solver_method},
                    {"setting": "jacobian_method", "value": least_squares_settings.jacobian_method},
                    {"setting": "ftol", "value": least_squares_settings.ftol},
                    {"setting": "xtol", "value": least_squares_settings.xtol},
                    {"setting": "gtol", "value": least_squares_settings.gtol},
                    {"setting": "rtol", "value": least_squares_settings.rtol},
                    {"setting": "x_scale", "value": "jac"},
                ]
            )

        if state_settings is not None:
            self.settings.extend(
                [
                    {"setting": "state_max_passes", "value": state_settings.max_passes},
                    {"setting": "state_tolerance", "value": state_settings.tolerance},
                ]
            )

        variable_labels = self._variable_labels(cache)
        for i, state in enumerate(cache.iteration_variables):
            label = variable_labels[i] if i < len(variable_labels) else f"x[{i}]"
            labels = cache.find_variable_labels(state)
            lower_bound = self._safe_float(getattr(state, "lower_bound", math.nan))
            upper_bound = self._safe_float(getattr(state, "upper_bound", math.nan))
            self.variable_metadata.append(
                {
                    "variable_index": i,
                    "variable": label,
                    "labels": "\n".join(labels),
                    "owner_kind": cache.iteration_items[i].owner_kind if i < len(cache.iteration_items) else "<unknown>",
                    "state_id": id(state),
                    "initial_value": self._safe_float(self.initial_x[i]) if self.initial_x is not None and i < len(self.initial_x) else self._safe_float(getattr(state, "value", math.nan)),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "has_lower_bound": math.isfinite(lower_bound),
                    "has_upper_bound": math.isfinite(upper_bound),
                    "has_bounds": bool(getattr(state, "has_bounds", False)),
                    "keep_feasible": bool(getattr(state, "keep_feasible", False)),
                }
            )

        try:
            residual_labels = cache.collect_residual_labels()
        except Exception:
            residual_labels = []

        for i, label in enumerate(residual_labels):
            self.residual_metadata.append(
                {
                    "residual_index": i,
                    "residual": label,
                    "owner": label.split(".residual", 1)[0],
                }
            )

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return math.nan

    @staticmethod
    def _numeric_value(value: Any) -> float | None:
        try:
            numeric = float(value)
        except Exception:
            return None

        if math.isnan(numeric) or math.isinf(numeric):
            return numeric

        return numeric

    @staticmethod
    def _plain_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, dict):
            return {
                SolverStatistics._plain_value(key): SolverStatistics._plain_value(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [SolverStatistics._plain_value(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    @staticmethod
    def _variable_labels(cache) -> list[str]:
        labels: list[str] = []

        for variable in cache.iteration_variables:
            labels.append("\n".join(cache.find_variable_labels(variable)))

        return labels

    @staticmethod
    def _residual_labels(cache, residual_count: int) -> list[str]:
        try:
            labels = cache.collect_residual_labels()
        except Exception:
            labels = []

        if len(labels) < residual_count:
            labels.extend(
                f"residual[{i}]"
                for i in range(len(labels), residual_count)
            )

        return labels


    def _ensure_residual_metadata(self, residual_labels: list[str]) -> None:
        known_indices = {row["residual_index"] for row in self.residual_metadata}

        for i, label in enumerate(residual_labels):
            if i in known_indices:
                continue

            self.residual_metadata.append(
                {
                    "residual_index": i,
                    "residual": label,
                    "owner": label.split(".residual", 1)[0],
                }
            )

    @staticmethod
    def _norms(values: np.ndarray, prefix: str) -> dict[str, Any]:
        if len(values) == 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_finite_count": 0,
                f"{prefix}_nonfinite_count": 0,
                f"{prefix}_l1_norm": 0.0,
                f"{prefix}_l2_norm": 0.0,
                f"{prefix}_inf_norm": 0.0,
                f"{prefix}_rms": 0.0,
                f"{prefix}_mean_abs": 0.0,
                f"{prefix}_median_abs": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_mean": 0.0,
            }

        finite = values[np.isfinite(values)]
        abs_values = np.abs(finite)

        if len(finite) == 0:
            return {
                f"{prefix}_count": int(len(values)),
                f"{prefix}_finite_count": 0,
                f"{prefix}_nonfinite_count": int(len(values)),
                f"{prefix}_l1_norm": math.inf,
                f"{prefix}_l2_norm": math.inf,
                f"{prefix}_inf_norm": math.inf,
                f"{prefix}_rms": math.inf,
                f"{prefix}_mean_abs": math.inf,
                f"{prefix}_median_abs": math.inf,
                f"{prefix}_min": math.nan,
                f"{prefix}_max": math.nan,
                f"{prefix}_mean": math.nan,
            }

        return {
            f"{prefix}_count": int(len(values)),
            f"{prefix}_finite_count": int(len(finite)),
            f"{prefix}_nonfinite_count": int(len(values) - len(finite)),
            f"{prefix}_l1_norm": float(np.sum(abs_values)),
            f"{prefix}_l2_norm": float(np.linalg.norm(finite)),
            f"{prefix}_inf_norm": float(np.max(abs_values)),
            f"{prefix}_rms": float(np.sqrt(np.mean(finite**2))),
            f"{prefix}_mean_abs": float(np.mean(abs_values)),
            f"{prefix}_median_abs": float(np.median(abs_values)),
            f"{prefix}_min": float(np.min(finite)),
            f"{prefix}_max": float(np.max(finite)),
            f"{prefix}_mean": float(np.mean(finite)),
        }

    def _record_network(self, evaluation: int, elapsed_time: float, cache) -> None:
        try:
            records = cache.network.save(filename=None, return_type="dict")
        except Exception as error:
            self.network_records.append(
                {
                    "evaluation": evaluation,
                    "time_s": elapsed_time,
                    "component_name": "<network>",
                    "component_type": "<error>",
                    "attribute": "save_error",
                    "value": f"{type(error).__name__}: {error}",
                    "numeric_value": None,
                }
            )
            return

        for record in records:
            value = self._plain_value(record.get("value"))
            self.network_records.append(
                {
                    "evaluation": evaluation,
                    "time_s": elapsed_time,
                    "component_name": record.get("component_name"),
                    "component_type": record.get("component_type"),
                    "attribute": record.get("attribute"),
                    "value": value,
                    "numeric_value": self._numeric_value(record.get("value")),
                }
            )

    def record(self, x: Any, residual: Any, cache, phase: str = "solver") -> None:
        if not self.enabled:
            return

        x = np.array(x, dtype=float)
        residual = np.array(residual, dtype=float)
        elapsed_time = time.perf_counter() - self.start_time
        delta_time = 0.0 if self.previous_time is None else elapsed_time - self.previous_time

        finite_residual = residual[np.isfinite(residual)]
        cost = 0.5 * float(np.dot(finite_residual, finite_residual)) if len(finite_residual) else math.inf if len(residual) else 0.0
        cost_change = 0.0 if self.previous_cost is None else cost - self.previous_cost
        cost_reduction = 0.0 if self.previous_cost is None else self.previous_cost - cost
        relative_cost_change = 0.0
        if self.previous_cost not in (None, 0.0) and math.isfinite(self.previous_cost):
            relative_cost_change = cost_change / abs(self.previous_cost)

        residual_labels = self._residual_labels(cache, len(residual))
        variable_labels = self._variable_labels(cache)
        self._ensure_residual_metadata(residual_labels)

        if len(residual) and not np.all(np.isnan(residual)):
            abs_residual = np.abs(residual)
            worst_index = int(np.nanargmax(abs_residual))
            best_index = int(np.nanargmin(abs_residual))
            worst_residual = residual_labels[worst_index]
            best_residual = residual_labels[best_index]
            worst_residual_value = float(residual[worst_index])
            best_residual_value = float(residual[best_index])
        else:
            worst_index = -1
            best_index = -1
            worst_residual = "<none>"
            best_residual = "<none>"
            worst_residual_value = 0.0
            best_residual_value = 0.0

        if self.previous_x is None or len(self.previous_x) != len(x):
            adjustment = np.zeros_like(x)
        else:
            adjustment = x - self.previous_x

        if self.initial_x is None or len(self.initial_x) != len(x):
            total_adjustment = np.zeros_like(x)
        else:
            total_adjustment = x - self.initial_x

        normalized_adjustment = np.abs(adjustment) / np.maximum(np.abs(x), 1.0)
        normalized_total_adjustment = np.abs(total_adjustment) / np.maximum(np.abs(x), 1.0)

        self.evaluation += 1
        evaluation = self.evaluation

        row = {
            "evaluation": evaluation,
            "phase": phase,
            "time_s": elapsed_time,
            "delta_time_s": delta_time,
            "cost": cost,
            "cost_change": cost_change,
            "cost_reduction": cost_reduction,
            "relative_cost_change": relative_cost_change,
            "worst_residual": worst_residual,
            "worst_residual_index": worst_index,
            "worst_residual_value": worst_residual_value,
            "worst_residual_abs_value": float(abs(worst_residual_value)),
            "best_residual": best_residual,
            "best_residual_index": best_index,
            "best_residual_value": best_residual_value,
            "best_residual_abs_value": float(abs(best_residual_value)),
            "max_abs_residual": float(np.nanmax(np.abs(residual))) if len(residual) and not np.all(np.isnan(residual)) else 0.0,
            "rms_residual": float(np.sqrt(np.nanmean(residual**2))) if len(residual) and not np.all(np.isnan(residual)) else 0.0,
            "max_variable_adjustment": float(np.nanmax(np.abs(adjustment))) if len(adjustment) else 0.0,
            "max_normalized_variable_adjustment": float(np.nanmax(normalized_adjustment)) if len(normalized_adjustment) else 0.0,
            "max_total_variable_adjustment": float(np.nanmax(np.abs(total_adjustment))) if len(total_adjustment) else 0.0,
            "max_normalized_total_variable_adjustment": float(np.nanmax(normalized_total_adjustment)) if len(normalized_total_adjustment) else 0.0,
        }
        row.update(self._norms(residual, "residual"))
        row.update(self._norms(x, "variable"))
        row.update(self._norms(adjustment, "step"))
        row.update(self._norms(total_adjustment, "total_step"))
        self.evaluations.append(row)

        for key, value in row.items():
            if key in {"evaluation", "phase"}:
                continue
            self.evaluation_metrics.append(
                {
                    "evaluation": evaluation,
                    "phase": phase,
                    "metric": key,
                    "value": value,
                }
            )

        max_abs_residual = row["max_abs_residual"]
        for i, value in enumerate(residual):
            label = residual_labels[i] if i < len(residual_labels) else f"residual[{i}]"
            abs_value = float(abs(value))
            self.residuals.append(
                {
                    "evaluation": evaluation,
                    "phase": phase,
                    "time_s": elapsed_time,
                    "delta_time_s": delta_time,
                    "residual_index": i,
                    "residual": label,
                    "owner": label.split(".residual", 1)[0],
                    "value": float(value),
                    "abs_value": abs_value,
                    "squared_value": float(value * value),
                    "normalized_by_max_abs_residual": abs_value / max_abs_residual if max_abs_residual else 0.0,
                    "is_worst_residual": i == worst_index,
                    "is_best_residual": i == best_index,
                    "is_finite": bool(math.isfinite(float(value))),
                }
            )

        for i, value in enumerate(x):
            label = variable_labels[i] if i < len(variable_labels) else f"x[{i}]"
            state = cache.iteration_variables[i] if i < len(cache.iteration_variables) else None
            lower_bound = self._safe_float(getattr(state, "lower_bound", math.nan)) if state is not None else math.nan
            upper_bound = self._safe_float(getattr(state, "upper_bound", math.nan)) if state is not None else math.nan
            dx = float(adjustment[i]) if i < len(adjustment) else 0.0
            ndx = float(normalized_adjustment[i]) if i < len(normalized_adjustment) else 0.0
            total_dx = float(total_adjustment[i]) if i < len(total_adjustment) else 0.0
            total_ndx = float(normalized_total_adjustment[i]) if i < len(normalized_total_adjustment) else 0.0
            value_float = float(value)
            self.variables.append(
                {
                    "evaluation": evaluation,
                    "phase": phase,
                    "time_s": elapsed_time,
                    "delta_time_s": delta_time,
                    "variable_index": i,
                    "variable": label,
                    "owner_kind": cache.iteration_items[i].owner_kind if i < len(cache.iteration_items) else "<unknown>",
                    "state_id": id(state) if state is not None else None,
                    "value": value_float,
                    "previous_value": float(self.previous_x[i]) if self.previous_x is not None and i < len(self.previous_x) else math.nan,
                    "initial_value": float(self.initial_x[i]) if self.initial_x is not None and i < len(self.initial_x) else math.nan,
                    "adjustment": dx,
                    "abs_adjustment": abs(dx),
                    "normalized_adjustment": ndx,
                    "total_adjustment": total_dx,
                    "abs_total_adjustment": abs(total_dx),
                    "normalized_total_adjustment": total_ndx,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "has_lower_bound": math.isfinite(lower_bound),
                    "has_upper_bound": math.isfinite(upper_bound),
                    "has_bounds": bool(getattr(state, "has_bounds", False)) if state is not None else False,
                    "keep_feasible": bool(getattr(state, "keep_feasible", False)) if state is not None else False,
                    "distance_to_lower_bound": value_float - lower_bound if math.isfinite(lower_bound) else math.inf,
                    "distance_to_upper_bound": upper_bound - value_float if math.isfinite(upper_bound) else math.inf,
                    "at_lower_bound": bool(math.isfinite(lower_bound) and value_float <= lower_bound + 1e-12),
                    "at_upper_bound": bool(math.isfinite(upper_bound) and value_float >= upper_bound - 1e-12),
                    "is_finite": bool(math.isfinite(value_float)),
                }
            )

        self._record_network(evaluation, elapsed_time, cache)

        self.previous_x = x.copy()
        self.previous_cost = cost
        self.previous_time = elapsed_time
        self.print_evaluation(self.evaluations[-1])

    def finalize(
        self,
        sol: Any,
        elapsed_time: float,
        cache,
        least_squares_settings=None,
        state_settings=None,
        overconstrained: bool = False,
    ) -> None:
        if not self.enabled:
            return

        final_residual = np.array(getattr(sol, "fun", []), dtype=float)
        final_x = np.array(getattr(sol, "x", []), dtype=float)
        final_row = {
            "success": bool(getattr(sol, "success", False)),
            "status": getattr(sol, "status", None),
            "message": str(getattr(sol, "message", "")),
            "solve_time_s": elapsed_time,
            "function_evaluations": getattr(sol, "nfev", None),
            "jacobian_evaluations": getattr(sol, "njev", None),
            "cost": getattr(sol, "cost", None),
            "optimality": getattr(sol, "optimality", None),
            "overconstrained": overconstrained,
        }
        final_row.update(self._norms(final_residual, "final_residual"))
        final_row.update(self._norms(final_x, "final_variable"))

        if least_squares_settings is not None:
            final_row.update(
                {
                    "solver_method": least_squares_settings.solver_method,
                    "jacobian_method": least_squares_settings.jacobian_method,
                    "ftol": least_squares_settings.ftol,
                    "xtol": least_squares_settings.xtol,
                    "gtol": least_squares_settings.gtol,
                    "rtol": least_squares_settings.rtol,
                }
            )

        if state_settings is not None:
            final_row.update(
                {
                    "state_max_passes": state_settings.max_passes,
                    "state_tolerance": state_settings.tolerance,
                }
            )

        active_mask = getattr(sol, "active_mask", None)
        if active_mask is not None:
            active_mask = np.array(active_mask, dtype=int)
            final_row["active_lower_bound_count"] = int(np.sum(active_mask < 0))
            final_row["active_upper_bound_count"] = int(np.sum(active_mask > 0))
            final_row["free_variable_count"] = int(np.sum(active_mask == 0))

            for i, value in enumerate(active_mask):
                if i < len(self.variables):
                    pass

        self.final_summary = [final_row]

    def print_evaluation(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self.console.print(
            f"call {row['evaluation']:>4d} | "
            f"{row['phase']} | "
            f"t={row['time_s']:.3f} s | "
            f"cost={row['cost']:.3e} | "
            f"max|r|={row['max_abs_residual']:.3e} | "
            f"rms={row['rms_residual']:.3e} | "
            f"max|dx|={row['max_variable_adjustment']:.3e} | "
            f"worst={row['worst_residual']} ({row['worst_residual_value']:.3e})"
        )

    def print_failure_report(
        self,
        max_rows: int = 10,
        residual: Any | None = None,
        residual_labels: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return

        if residual is not None:
            residual = np.array(residual, dtype=float)
            residual_labels = residual_labels or []

            if len(residual_labels) < len(residual):
                residual_labels.extend(
                    f"residual[{i}]"
                    for i in range(len(residual_labels), len(residual))
                )

            rows = [
                {
                    "residual": residual_labels[i],
                    "value": float(value),
                    "abs_value": float(abs(value)),
                }
                for i, value in enumerate(residual)
            ]
        else:
            if not self.residuals:
                return

            last_evaluation = self.evaluations[-1]["evaluation"]
            rows = [
                row for row in self.residuals
                if row["evaluation"] == last_evaluation
            ]

        rows = sorted(rows, key=lambda row: row["abs_value"], reverse=True)[:max_rows]

        table = Table(
            title="Largest Final Residuals",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Residual", style="#fdf0d5")
        table.add_column("Value", justify="right", style="#3B629E")
        table.add_column("Abs Value", justify="right", style="#D84135")

        for row in rows:
            table.add_row(
                _plain(row["residual"]),
                _plain(f"{row['value']:.6e}"),
                _plain(f"{row['abs_value']:.6e}"),
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _table_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "evaluations": self.evaluations,
            "evaluation_metrics": self.evaluation_metrics,
            "residuals": self.residuals,
            "variables": self.variables,
            "network": self.network_records,
            "variable_metadata": self.variable_metadata,
            "residual_metadata": self.residual_metadata,
            "settings": self.settings,
            "final_summary": self.final_summary,
        }

    @staticmethod
    def _wide_table(rows: list[dict[str, Any]], index: str, columns: str, values: str) -> list[dict[str, Any]]:
        if not rows:
            return []

        wide: dict[Any, dict[str, Any]] = {}
        column_order: list[str] = []

        for row in rows:
            if index not in row or columns not in row:
                continue

            index_value = row[index]
            column_name = str(row[columns])

            if column_name not in column_order:
                column_order.append(column_name)

            if index_value not in wide:
                wide[index_value] = {index: index_value}

            wide[index_value][column_name] = row.get(values)

        return [
            {column: row.get(column) for column in [index] + column_order}
            for row in wide.values()
        ]

    def export(self, filename: str | HDF5Target | None) -> None:
        if not self.enabled or filename is None:
            return

        tables = self._table_dict()
        tables["residuals_wide"] = self._wide_table(
            tables["residuals"],
            "evaluation",
            "residual",
            "value",
        )
        tables["variables_wide"] = self._wide_table(
            tables["variables"],
            "evaluation",
            "variable",
            "value",
        )
        tables["network_numeric_wide"] = self._wide_table(
            tables["network"],
            "evaluation",
            "attribute",
            "numeric_value",
        )

        write_tables(filename, tables, group_path="diagnostics/steady_state/current", kind="steady_state_statistics")


def statistics_path(filename: str) -> HDF5Target:
    return HDF5Target(filename, "diagnostics/steady_state/current")


def model_option_statistics_path(filename: str, model_name: str, option_name: str) -> HDF5Target:
    return HDF5Target(
        filename,
        f"diagnostics/model_options/{safe_group_name(model_name)}/{safe_group_name(option_name)}/statistics",
    )
