"""Rich terminal diagnostics for steady-state runs.

All printing lives here so numerical solver code can remain UI-free. The class
uses ``RuntimeCache`` only through a getter, which keeps labels and residual
counts current after model options swap components in and out.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from .models import ModelFailure


class SteadyStatePrinter:
    """Build and print the verbose tables used by ``SteadyState``.

    The printer has no control-flow authority. It only reads the network/cache
    after a static evaluation or solve has succeeded.
    """

    def __init__(self, network, cache_getter: Callable[[], Any], console: Console | None = None) -> None:
        self.network = network
        self._cache_getter = cache_getter
        self.console = console or Console()

    def print_solution(self) -> None:
        """Print the exported network state table."""
        records = self.network.save(return_type="dict")
        table = Table(
            title=f"{self.network.name} Solution",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Component", style="#D84135", no_wrap=True)
        table.add_column("Type", style="#3B629E", no_wrap=True)
        table.add_column("Attribute", style="#fdf0d5", no_wrap=True)
        table.add_column("Value", justify="right")

        for record in records:
            value = record["value"]
            value_text = f"{value:.6g}" if isinstance(value, float) else str(value)
            if value_text == "<uninitialized>":
                value_text = "[dim]<uninitialized>[/dim]"
            elif value_text == "<unavailable>":
                value_text = "[red]<unavailable>[/red]"

            table.add_row(
                str(record["component_name"]),
                str(record["component_type"]),
                str(record["attribute"]),
                value_text,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def print_model_failures(self, failures: list[ModelFailure]) -> None:
        """Print model options skipped after evaluation failures."""
        if not failures:
            return

        table = Table(
            title="Skipped Model Options",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Model", style="#D84135", no_wrap=True)
        table.add_column("Option", style="#fdf0d5", no_wrap=True)
        table.add_column("Error Type", style="#3B629E", no_wrap=True)
        table.add_column("Error")

        for failure in failures:
            table.add_row(
                failure.model,
                failure.option,
                failure.error_type,
                failure.error,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def print_static(self, elapsed_time: float) -> None:
        """Print summary information for a static evaluation."""
        cache = self._cache_getter()
        try:
            residual_count = len(cache.collect_residuals())
        except Exception:
            residual_count = 0

        table = Table(
            title="Static Network Evaluation",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Quantity", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Mode", "Static evaluation")
        table.add_row("Nonlinear solve", "Not performed")
        table.add_row("Evaluation time", f"{elapsed_time:.3f} s")
        table.add_row("Components", str(len(cache.component_list)))
        table.add_row("Iteration variables", str(len(cache.iteration_variables)))
        table.add_row("Residuals", str(residual_count))

        self.console.print()
        self.console.print(table)
        self.console.print()

    def print_solve(
        self,
        sol: Any,
        x0: np.ndarray,
        method: str,
        jac: str,
        ftol: float,
        xtol: float,
        gtol: float,
        rtol: float,
        overconstrained: bool = False,
        elapsed_time: float = 0.0,
    ) -> None:
        """Print solver convergence, variable, and residual tables."""
        cache = self._cache_getter()
        max_residual = np.max(np.abs(sol.fun)) if len(sol.fun) else 0.0
        rms_residual = np.sqrt(np.mean(sol.fun**2)) if len(sol.fun) else 0.0
        dx = np.array(sol.x, dtype=float) - np.array(x0, dtype=float)
        normalized_dx = np.abs(dx) / np.maximum(np.abs(sol.x), 1.0)
        max_normalized_dx = np.max(normalized_dx) if len(normalized_dx) else 0.0

        summary = Table(
            title="Steady-State Solver Summary",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        summary.add_column("Quantity", style="bold")
        summary.add_column("Value", justify="right")
        summary.add_row("Success", str(sol.success))
        summary.add_row("Status", str(sol.status))
        summary.add_row("Message", str(sol.message))
        if overconstrained:
            summary.add_row("Warning", "System is overconstrained", style="yellow")
        summary.add_row("Solver method", method)
        summary.add_row("Jacobian method", jac)
        summary.add_row("Solve time", f"{elapsed_time:.3f} s")
        summary.add_row("Function evaluations", str(sol.nfev))
        if getattr(sol, "njev", None) is not None:
            summary.add_row("Jacobian evaluations", str(sol.njev))
        if hasattr(sol, "cost"):
            summary.add_row("Cost", f"{sol.cost:.6e}")
        if hasattr(sol, "optimality"):
            summary.add_row("Optimality", f"{sol.optimality:.3e}")
        summary.add_row("Max |residual|", f"{max_residual:.3e}")
        summary.add_row("RMS residual", f"{rms_residual:.3e}")
        summary.add_row("Max normalized variable adjustment", f"{max_normalized_dx:.3e}")
        summary.add_row("Residual tolerance", f"{rtol:.3e}")
        summary.add_row("ftol", f"{ftol:.3e}")
        summary.add_row("xtol", f"{xtol:.3e}")
        summary.add_row("gtol", f"{gtol:.3e}")

        variables = Table(
            title="Solution Variables",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        variables.add_column("Index", justify="right", style="dim")
        variables.add_column("Variable", style="#fdf0d5")
        variables.add_column("Value", justify="right", style="#D84135")
        variables.add_column("Variable Adjustment", justify="right", style="#3B629E")
        variables.add_column("Normalized Variable Adjustment", justify="right", style="#3B629E")

        variable_labels = [cache.find_variable_labels(var) for var in cache.iteration_variables]
        for i, value in enumerate(sol.x):
            label = "\n".join(variable_labels[i]) if i < len(variable_labels) else "<unlabeled>"
            variables.add_row(
                f"x[{i}]",
                label,
                f"{value:.6e}",
                f"{dx[i]:+.6e}",
                f"{normalized_dx[i]:.3e}",
            )

        residuals = Table(
            title="Residuals",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        residuals.add_column("Index", justify="right", style="dim")
        residuals.add_column("Residual", style="#fdf0d5")
        residuals.add_column("Value", justify="right", style="#3B629E")

        try:
            residual_labels = cache.collect_residual_labels()
        except Exception:
            residual_labels = []

        for i, value in enumerate(sol.fun):
            label = residual_labels[i] if i < len(residual_labels) else "<unlabeled>"
            residuals.add_row(f"r[{i}]", label, f"{value:.6e}")

        self.console.print()
        self.console.print(summary)
        self.console.print(variables)
        self.console.print(residuals)
        self.console.print()
