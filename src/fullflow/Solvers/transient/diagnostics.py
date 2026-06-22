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


def _plain(value: Any) -> Text:
    """Return literal Rich text so user labels are never parsed as markup."""
    return Text(str(value))


class TransientPrinter:
    """Build and print transient solver diagnostics."""

    def __init__(self, network, cache_getter: Callable[[], Any], console: Console | None = None) -> None:
        self.network = network
        self._cache_getter = cache_getter
        self.console = console or Console()
        self._steady_printer = SteadyStatePrinter(network, cache_getter, self.console)

    def print_step(self, diagnostics) -> None:
        """Print one compact accepted-step progress line."""
        self.console.print(
            f"Transient step accepted: "
            f"t={diagnostics.time:.9g}, "
            f"dt={diagnostics.dt:.9g}, "
            f"max|residual|={diagnostics.max_residual:.3e}"
        )

    def print_summary(
        self,
        *,
        diagnostics_list: list[Any],
        start_time: float,
        final_time: float,
        requested_dt: float,
        method: str,
        jac: str,
        ftol: float,
        xtol: float,
        gtol: float,
        rtol: float,
        solve_time: float,
    ) -> None:
        """Print a final transient summary similar to steady-state verbose."""
        final = diagnostics_list[-1] if diagnostics_list else None
        final_residual = final.residual if final is not None else np.array([], dtype=float)
        max_residual = np.max(np.abs(final_residual)) if len(final_residual) else 0.0
        rms_residual = np.sqrt(np.mean(final_residual**2)) if len(final_residual) else 0.0
        total_nfev = 0
        total_njev = 0
        total_cost = 0.0
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
        table.add_row(_plain("Success"), _plain(True))
        if last_status is not None:
            table.add_row(_plain("Last SciPy status"), _plain(last_status))
        table.add_row(_plain("Last SciPy message"), _plain(last_message))
        table.add_row(_plain("Solver type"), _plain("Fixed-step implicit backward Euler"))
        table.add_row(_plain("Nonlinear solver"), _plain("scipy.optimize.least_squares"))
        table.add_row(_plain("Solver method"), _plain(method))
        table.add_row(_plain("Jacobian method"), _plain(jac))
        table.add_row(_plain("Start time"), _plain(f"{start_time:.9g}"))
        table.add_row(_plain("Final time"), _plain(f"{final_time:.9g}"))
        table.add_row(_plain("Requested dt"), _plain(f"{requested_dt:.9g}"))
        table.add_row(_plain("Accepted steps"), _plain(len(diagnostics_list)))
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
        table.add_row(_plain("gtol"), _plain(f"{gtol:.3e}"))

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

    def print_network_solution(self) -> None:
        """Print the final network state using the steady-state table format."""
        self._steady_printer.print_network_solution()
