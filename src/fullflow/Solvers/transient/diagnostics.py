"""Rich terminal diagnostics for transient runs.

``verbose`` and ``statistics`` intentionally mean different things:

* ``verbose=True`` prints one final solver summary and the final network state,
  matching the steady-state user experience.
* ``statistics=True`` prints the accepted timestep progression as the solve runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from fullflow.Solvers.steady_state.diagnostics import SteadyStatePrinter


def _plain(value: Any, style: str | None = None) -> Text:
    """Return literal Rich text so user labels are never parsed as markup."""
    return Text(str(value), style=style)


SENSOR_ROLE_STYLES = {
    "greenline": "#2ECC71",
    "blueline": "#4EA1FF",
    "yellowline": "#FFD43B",
    "redline": "#FF4D6D",
}

SENSOR_ROLE_BOLD_STYLES = {
    "greenline": "bold #2ECC71",
    "blueline": "bold #4EA1FF",
    "yellowline": "bold #FFD43B",
    "redline": "bold #FF4D6D",
}


def _sensor_role(value: Any) -> str:
    return str(value or "event").lower()


def _sensor_role_text(value: Any, *, upper: bool = True) -> Text:
    role = _sensor_role(value)
    label = role.upper() if upper else role
    return _plain(label, SENSOR_ROLE_BOLD_STYLES.get(role, "bold"))


def _sensor_action_text(value: Any, role: str | None = None) -> Text:
    action = str(value or "")
    style = SENSOR_ROLE_BOLD_STYLES.get(_sensor_role(role)) if role else None
    return _plain(action, style)


def _sensor_crossing_text(value: Any, role: str | None = None) -> Text:
    crossing = str(value or "")
    style = SENSOR_ROLE_STYLES.get(_sensor_role(role)) if role else None
    return _plain(crossing, style)


def _sensor_value_text(value: Any, role: str, fmt: str = ".6g") -> Text:
    try:
        text = format(float(value), fmt)
    except Exception:
        text = str(value)
    return _plain(text, SENSOR_ROLE_STYLES.get(_sensor_role(role)))


def _sensor_line_style(event: Any) -> str | None:
    return SENSOR_ROLE_STYLES.get(_sensor_role(getattr(event, "role", "event")))


def _sensor_line_bold_style(event: Any) -> str | None:
    return SENSOR_ROLE_BOLD_STYLES.get(_sensor_role(getattr(event, "role", "event")))


class TransientPrinter:
    """Build and print transient solver diagnostics."""

    def __init__(self, network, cache_getter: Callable[[], Any], console: Console | None = None) -> None:
        self.network = network
        self._cache_getter = cache_getter
        self.console = console or Console()
        self._steady_printer = SteadyStatePrinter(network, cache_getter, self.console)

    def print_step(self, diagnostics) -> None:
        """Print one compact accepted-step progress line."""
        retry_text = f", retries={diagnostics.retries}" if diagnostics.retries else ""
        self.console.print(
            f"Transient step accepted: "
            f"t={diagnostics.time:.9g}, "
            f"dt={diagnostics.dt:.9g}, "
            f"max|residual|={diagnostics.max_residual:.3e}"
            f"{retry_text}"
        )

    def print_summary(
        self,
        *,
        diagnostics_list: list[Any],
        start_time: float,
        final_time: float,
        requested_dt: Any,
        method: str,
        jac: str,
        ftol: float,
        xtol: float,
        gtol: float | None,
        rtol: float,
        solve_time: float,
        success: bool = True,
        stop_reason: str | None = None,
    ) -> None:
        """Print a final transient summary similar to steady-state verbose."""
        final = diagnostics_list[-1] if diagnostics_list else None
        final_residual = final.residual if final is not None else np.array([], dtype=float)
        max_residual = np.max(np.abs(final_residual)) if len(final_residual) else 0.0
        rms_residual = np.sqrt(np.mean(final_residual**2)) if len(final_residual) else 0.0
        total_nfev = 0
        total_njev = 0
        total_cost = 0.0
        total_retries = sum(int(getattr(row, "retries", 0) or 0) for row in diagnostics_list)
        min_dt = min((float(getattr(row, "dt", 0.0) or 0.0) for row in diagnostics_list), default=0.0)
        max_dt = max((float(getattr(row, "dt", 0.0) or 0.0) for row in diagnostics_list), default=0.0)
        last_status = None
        last_message = "No nonlinear timestep solve was required."
        last_optimality = None

        for row in diagnostics_list:
            sol = row.sol
            if sol is None:
                continue
            total_nfev += int(getattr(sol, "nfev", 0) or 0)
            total_njev += int(getattr(sol, "njev", 0) or 0)
            total_cost += float(getattr(sol, "cost", 0.0) or 0.0)
            last_status = getattr(sol, "status", None)
            last_message = str(getattr(sol, "message", ""))
            last_optimality = getattr(sol, "optimality", None)

        table = Table(
            title="Transient Solver Summary",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Quantity", style="bold")
        table.add_column("Value", justify="right")
        table.add_row(_plain("Success"), _plain(success))
        if stop_reason:
            table.add_row(_plain("Stop reason"), _plain(stop_reason))
        if last_status is not None:
            table.add_row(_plain("Last SciPy status"), _plain(last_status))
        table.add_row(_plain("Last SciPy message"), _plain(last_message))
        requested_dt_text = f"{float(requested_dt):.9g}"

        table.add_row(_plain("Solver type"), _plain("Fixed-step implicit backward Euler"))
        table.add_row(_plain("Nonlinear solver"), _plain("scipy.optimize.least_squares"))
        table.add_row(_plain("Solver method"), _plain(method))
        table.add_row(_plain("Jacobian method"), _plain(jac))
        table.add_row(_plain("Start time"), _plain(f"{start_time:.9g}"))
        table.add_row(_plain("Final time"), _plain(f"{final_time:.9g}"))
        table.add_row(_plain("Requested dt"), _plain(requested_dt_text))
        table.add_row(_plain("Accepted steps"), _plain(len(diagnostics_list)))
        table.add_row(_plain("Automatic retries"), _plain(total_retries))
        if diagnostics_list:
            table.add_row(_plain("Minimum dt used"), _plain(f"{min_dt:.9g}"))
            table.add_row(_plain("Maximum dt used"), _plain(f"{max_dt:.9g}"))
        table.add_row(_plain("Solve time"), _plain(f"{solve_time:.3f} s"))
        table.add_row(_plain("Total function evaluations"), _plain(total_nfev))
        table.add_row(_plain("Total Jacobian evaluations"), _plain(total_njev))
        table.add_row(_plain("Total cost"), _plain(f"{total_cost:.6e}"))
        if last_optimality is not None:
            table.add_row(_plain("Last optimality"), _plain(f"{float(last_optimality):.3e}"))
        table.add_row(_plain("Final max |residual|"), _plain(f"{max_residual:.3e}"))
        table.add_row(_plain("Final RMS residual"), _plain(f"{rms_residual:.3e}"))
        table.add_row(_plain("Residual tolerance"), _plain(f"{rtol:.3e}"))
        table.add_row(_plain("ftol"), _plain(f"{ftol:.3e}"))
        table.add_row(_plain("xtol"), _plain(f"{xtol:.3e}"))
        table.add_row(_plain("gtol"), _plain("disabled" if gtol is None else f"{gtol:.3e}"))

        self.console.print()
        self.console.print(table)

        variables = Table(
            title="Final Transient Variables",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        variables.add_column("Index", justify="right", style="dim")
        variables.add_column("Solver Variable", style="#fdf0d5")
        variables.add_column("Integrated State", style="#fdf0d5")
        variables.add_column("Variable Value", justify="right", style="#D84135")
        variables.add_column("State Value", justify="right", style="#D84135")
        variables.add_column("State Previous", justify="right", style="#3B629E")

        cache = self._cache_getter()
        for i, item in enumerate(cache.transient_items):
            try:
                previous = item.state.previous
            except Exception:
                previous = "<unavailable>"
            variables.add_row(
                _plain(f"x[{i}]"),
                _plain(item.variable_label),
                _plain(item.state_label),
                _plain(f"{float(item.variable.value):.6e}"),
                _plain(f"{float(item.state.value):.6e}"),
                _plain(previous if isinstance(previous, str) else f"{float(previous):.6e}"),
            )

        if len(cache.transient_items):
            self.console.print(variables)

        residuals = Table(
            title="Final Residuals",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        residuals.add_column("Index", justify="right", style="dim")
        residuals.add_column("Residual", style="#fdf0d5")
        residuals.add_column("Value", justify="right", style="#3B629E")

        labels = cache.collect_residual_labels()
        for i, value in enumerate(final_residual):
            label = labels[i] if i < len(labels) else f"residual[{i}]"
            residuals.add_row(_plain(f"r[{i}]"), _plain(label), _plain(f"{value:.6e}"))

        self.console.print(residuals)
        self.console.print()


    def _print_sensor_event_banner(self, event: Any, *, abort: bool = False, filename: str | None = None) -> None:
        """Print one high-visibility Sensor event message in its role color."""
        role = _sensor_role(getattr(event, "role", "event"))
        style = SENSOR_ROLE_BOLD_STYLES.get(role, "bold")
        line = "=" * 72
        role_label = role.upper()
        title = f"{role_label} ABORT" if abort else f"{role_label} EVENT"

        self.console.print()
        self.console.print(_plain(line, style))
        self.console.print(_plain(title, style))
        self.console.print(_plain(line, style))
        self.console.print(_plain(f"Time:     {float(getattr(event, 'time', 0.0)):.9g} s", style))
        self.console.print(_plain(f"Sensor:   {getattr(event, 'sensor', '')}", style))
        self.console.print(_plain(f"Line:     {getattr(event, 'trace', '')}", style))
        self.console.print(_plain(f"Value:    {float(getattr(event, 'reading', float('nan'))):.9g}", style))
        self.console.print(_plain(f"Limit:    {float(getattr(event, 'line_value', float('nan'))):.9g}", style))
        self.console.print(_plain(f"Crossing: {getattr(event, 'crossing_direction', '')}", style))
        self.console.print(_plain(f"Action:   {getattr(event, 'action', '')}", style))
        if abort:
            exported = "No HDF5 filename was provided."
            if filename:
                exported = f"Partial results will be exported to {filename}."
            self.console.print(_plain("Transient stopped because a redline was crossed.", style))
            self.console.print(_plain(exported, style))
        self.console.print(_plain(line, style))
        self.console.print()

    def print_sensor_events(self, events: list[Any]) -> None:
        """Print sensor condition events that occurred during a timestep."""
        if not events:
            return

        for event in events:
            self._print_sensor_event_banner(event)

    def print_redline_abort(self, event: Any, filename: str | None = None) -> None:
        """Backward-compatible alias for printing a redline Sensor event."""
        self._print_sensor_event_banner(event, abort=False, filename=filename)

    def print_sequence_abort(self, record: dict[str, Any]) -> None:
        """Print a clean Sequence abort banner."""
        style = "bold red"
        line = "=" * 72
        self.console.print()
        self.console.print(_plain(line, style))
        self.console.print(_plain("SEQUENCE ABORT", style))
        self.console.print(_plain(line, style))
        self.console.print(_plain(f"Time:      {float(record.get('time', 0.0)):.9g} s", style))
        self.console.print(_plain(f"Sequence:  {record.get('sequence', '')}", style))
        self.console.print(_plain(f"Condition: {record.get('condition', '')}", style))
        self.console.print(_plain(f"Delay:     {float(record.get('delay', 0.0)):.9g} s", style))
        message = str(record.get("message", "Sequence abort requested."))
        self.console.print(_plain(message, style))
        self.console.print(_plain("Transient stopped cleanly. HDF5 export will still be written if filename was provided.", style))
        self.console.print(_plain(line, style))
        self.console.print()

    def print_sensor_event_summary(self, events: list[Any]) -> None:
        """Print final sensor condition event details with simple per-row colors."""
        if not events:
            return

        details = Table(
            title="Sensor Event Details",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        details.add_column("Role", style="bold")
        details.add_column("Time", justify="right")
        details.add_column("Sensor")
        details.add_column("Condition")
        details.add_column("Reading", justify="right")
        details.add_column("Line", justify="right")
        details.add_column("Crossing")
        details.add_column("Action")

        for event in events:
            role = _sensor_role(getattr(event, "role", "event"))
            row_style = SENSOR_ROLE_STYLES.get(role)
            details.add_row(
                _plain(role.upper(), row_style),
                _plain(f"{float(getattr(event, 'time', 0.0)):.9g}", row_style),
                _plain(getattr(event, "sensor", ""), row_style),
                _plain(getattr(event, "trace", ""), row_style),
                _sensor_value_text(getattr(event, "reading", float("nan")), role),
                _sensor_value_text(getattr(event, "line_value", float("nan")), role),
                _sensor_crossing_text(getattr(event, "crossing_direction", ""), role),
                _sensor_action_text(getattr(event, "action", ""), role),
            )

        self.console.print()
        self.console.print(details)
        self.console.print()

    def print_model_failures(self, failures: list[Any]) -> None:
        """Print model-option failures using the steady-state table format."""
        self._steady_printer.print_model_failures(failures)

    def print_network_solution(self) -> None:
        """Print the final network state using the steady-state table format."""
        self._steady_printer.print_network_solution()
