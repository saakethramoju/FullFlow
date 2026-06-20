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
from rich.text import Text

from .models import ModelFailure


def _plain(value: Any) -> Text:
    """Return literal Rich text so user labels are never parsed as markup."""
    return Text(str(value))


def _styled(value: Any, style: str) -> Text:
    """Return literal Rich text with style, without parsing markup tags."""
    return Text(str(value), style=style)


def _format_number(value: Any) -> str:
    """Format scalar numeric values compactly for terminal tables."""
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            return str(value)
        return f"{value:.6g}"

    return str(value)


def _format_array(value: np.ndarray) -> str:
    """Summarize arrays instead of printing every element."""
    array = np.asarray(value)

    if array.size == 0:
        return f"array(shape={array.shape}, empty)"

    if array.dtype.kind in {"b", "i", "u", "f", "c"}:
        try:
            return (
                f"array(shape={array.shape}, "
                f"min={_format_number(np.nanmin(array))}, "
                f"max={_format_number(np.nanmax(array))})"
            )
        except Exception:
            pass

    return f"array(shape={array.shape}, dtype={array.dtype})"


def _format_dict(value: dict[Any, Any], *, depth: int = 0, max_items: int = 6) -> str:
    """Format dictionaries compactly for display."""
    if not value:
        return "{}"

    pieces = []
    items = list(value.items())

    for key, item in items[:max_items]:
        pieces.append(f"{key}={_format_value(item, depth=depth + 1)}")

    if len(items) > max_items:
        pieces.append("...")

    return "{" + ", ".join(pieces) + "}"


def _is_stream_tuple(value: Any) -> bool:
    """Return True for a Composition-style ``(mass_flow, composition)`` tuple."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and (isinstance(value[1], dict) or value[1] is None)
    )


def _format_streams(value: list[Any] | tuple[Any, ...], *, depth: int = 0) -> str:
    """Format lists of ``(mass_flow, composition)`` stream tuples."""
    lines = []

    for index, stream in enumerate(value):
        mass_flow, composition = stream

        if composition is None:
            composition_text = "None"
        else:
            composition_text = _format_dict(composition, depth=depth + 1)

        lines.append(
            f"[{index}] mdot={_format_value(mass_flow, depth=depth + 1)}, "
            f"x={composition_text}"
        )

    return "\n".join(lines)


def _format_sequence(
    value: list[Any] | tuple[Any, ...],
    *,
    depth: int = 0,
    max_items: int = 6,
) -> str:
    """Format lists and tuples compactly for display."""
    if not value:
        return "[]" if isinstance(value, list) else "()"

    if all(_is_stream_tuple(item) for item in value):
        return _format_streams(value, depth=depth)

    items = list(value)
    pieces = [
        _format_value(item, depth=depth + 1)
        for item in items[:max_items]
    ]

    if len(items) > max_items:
        pieces.append("...")

    open_bracket, close_bracket = ("[", "]") if isinstance(value, list) else ("(", ")")
    return open_bracket + ", ".join(pieces) + close_bracket


def _format_value(value: Any, *, depth: int = 0) -> str:
    """Format values for verbose terminal output.

    This is intentionally display-only. It should not be used for saving,
    exporting, or numerical work.
    """
    if isinstance(value, str) and value in {"<uninitialized>", "<unavailable>"}:
        return value

    if isinstance(value, np.ndarray):
        return _format_array(value)

    if isinstance(value, dict):
        return _format_dict(value, depth=depth)

    if isinstance(value, (list, tuple)):
        return _format_sequence(value, depth=depth)

    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return _format_number(value)

    return str(value)


class SteadyStatePrinter:
    """Build and print the verbose tables used by ``SteadyState``.

    The printer has no control-flow authority. It only reads the network/cache
    after a static evaluation or solve has succeeded.
    """

    def __init__(self, network, cache_getter: Callable[[], Any], console: Console | None = None) -> None:
        self.network = network
        self._cache_getter = cache_getter
        self.console = console or Console()

    def print_network_solution(self) -> None:
        """Print the exported network state table."""
        records = self.network.save(return_type="dict")
        table = Table(
            title=_plain(f"{self.network.name} Solution"),
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Component", style="#D84135", no_wrap=True)
        table.add_column("Type", style="#3B629E", no_wrap=True)
        table.add_column("Attribute", style="#fdf0d5", no_wrap=True)
        table.add_column("Value", overflow="fold", max_width=90)

        for record in records:
            value_text = _format_value(record["value"])

            if value_text == "<uninitialized>":
                value_cell = _styled(value_text, "dim")
            elif value_text == "<unavailable>":
                value_cell = _styled(value_text, "red")
            else:
                value_cell = _plain(value_text)

            table.add_row(
                _plain(record["component_name"]),
                _plain(record["component_type"]),
                _plain(record["attribute"]),
                value_cell,
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
                _plain(failure.model),
                _plain(failure.option),
                _plain(failure.error_type),
                _plain(failure.error),
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
        table.add_row(_plain("Mode"), _plain("Static evaluation"))
        table.add_row(_plain("Nonlinear solve"), _plain("Not performed"))
        table.add_row(_plain("Evaluation time"), _plain(f"{elapsed_time:.3f} s"))
        table.add_row(_plain("Components"), _plain(len(cache.component_list)))
        table.add_row(_plain("Iteration variables"), _plain(len(cache.iteration_variables)))
        table.add_row(_plain("Residuals"), _plain(residual_count))

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
        summary.add_row(_plain("Success"), _plain(sol.success))
        summary.add_row(_plain("Status"), _plain(sol.status))
        summary.add_row(_plain("Message"), _plain(sol.message))
        if overconstrained:
            summary.add_row(_plain("Warning"), _plain("System is overconstrained"), style="yellow")
        summary.add_row(_plain("Solver method"), _plain(method))
        summary.add_row(_plain("Jacobian method"), _plain(jac))
        summary.add_row(_plain("Solve time"), _plain(f"{elapsed_time:.3f} s"))
        summary.add_row(_plain("Function evaluations"), _plain(sol.nfev))
        if getattr(sol, "njev", None) is not None:
            summary.add_row(_plain("Jacobian evaluations"), _plain(sol.njev))
        if hasattr(sol, "cost"):
            summary.add_row(_plain("Cost"), _plain(f"{sol.cost:.6e}"))
        if hasattr(sol, "optimality"):
            summary.add_row(_plain("Optimality"), _plain(f"{sol.optimality:.3e}"))
        summary.add_row(_plain("Max |residual|"), _plain(f"{max_residual:.3e}"))
        summary.add_row(_plain("RMS residual"), _plain(f"{rms_residual:.3e}"))
        summary.add_row(_plain("Max normalized variable adjustment"), _plain(f"{max_normalized_dx:.3e}"))
        summary.add_row(_plain("Residual tolerance"), _plain(f"{rtol:.3e}"))
        summary.add_row(_plain("ftol"), _plain(f"{ftol:.3e}"))
        summary.add_row(_plain("xtol"), _plain(f"{xtol:.3e}"))
        summary.add_row(_plain("gtol"), _plain(f"{gtol:.3e}"))

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
                _plain(f"x[{i}]"),
                _plain(label),
                _plain(f"{value:.6e}"),
                _plain(f"{dx[i]:+.6e}"),
                _plain(f"{normalized_dx[i]:.3e}"),
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
            residuals.add_row(_plain(f"r[{i}]"), _plain(label), _plain(f"{value:.6e}"))

        self.console.print()
        self.console.print(summary)
        self.console.print(variables)
        self.console.print(residuals)
        self.console.print()
