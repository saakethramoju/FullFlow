from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _text(value, style=None):
    return Text(str(value), style=style)


def _summary(result):
    table = Table(
        title=f"Linearized State-Space Model at t = {result.time:g} s",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Quantity", style="bold")
    table.add_column("Value")
    table.add_row("State equation", "δx_dot = A δx + B δu")
    table.add_row("Output equation", "δy = C δx + D δu")
    table.add_row("States", str(len(result.states)))
    table.add_row("Inputs", str(len(result.inputs)))
    table.add_row("Outputs", str(len(result.outputs)))
    return table


def _variables(result, width):
    table = Table(
        title="State-Space Variables",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Index", justify="right", style="dim", no_wrap=True)
    table.add_column("Kind", style="#3B629E", no_wrap=True)
    table.add_column(
        "Variable",
        style="#fdf0d5",
        no_wrap=True,
        overflow="ellipsis",
        max_width=max(16, width - 57),
    )
    table.add_column("Operating Point", justify="right", style="#D84135", no_wrap=True)
    table.add_column("Derivative", justify="right", style="#3B629E", no_wrap=True)

    for index, (name, value, derivative) in enumerate(zip(result.states, result.x0, result.xdot0)):
        table.add_row(f"x[{index}]", "State", _text(name), f"{value:.6e}", f"{derivative:.6e}")
    for index, (name, value) in enumerate(zip(result.inputs, result.u0)):
        table.add_row(f"u[{index}]", "Input", _text(name), f"{value:.6e}", "—")
    for index, (name, value) in enumerate(zip(result.outputs, result.y0)):
        table.add_row(f"y[{index}]", "Output", _text(name), f"{value:.6e}", "—")
    return table


def _matrix_tables(name, matrix, row_symbol, column_symbol, width):
    rows, columns = matrix.shape
    block_size = max(1, (max(40, width) - 14) // 16)
    blocks = [(0, 0)] if columns == 0 else [
        (start, min(columns, start + block_size))
        for start in range(0, columns, block_size)
    ]

    tables = []
    for start, stop in blocks:
        suffix = "" if len(blocks) == 1 else f" — {column_symbol}[{start}:{stop}]"
        title = f"{name} Matrix ({rows} × {columns}){suffix}"
        table = Table(
            title=title,
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            min_width=min(width, len(title) + 2),
        )
        table.add_column("Row", justify="right", style="dim", no_wrap=True)
        for column in range(start, stop):
            table.add_column(
                f"{column_symbol}[{column}]",
                justify="right",
                style="#3B629E",
                no_wrap=True,
            )
        if rows == 0 or columns == 0:
            table.add_row("—", *(["—"] * (stop - start)))
        else:
            for row in range(rows):
                table.add_row(
                    f"{row_symbol}[{row}]",
                    *(f"{matrix[row, column]:.6e}" for column in range(start, stop)),
                )
        tables.append(table)
    return tables


def print_linearization(result, *, console=None):
    console = console or Console()
    console.print()
    console.print(_summary(result))
    console.print(_variables(result, console.width))
    for name, matrix, rows, columns in (
        ("A", result.A, "x_dot", "x"),
        ("B", result.B, "x_dot", "u"),
        ("C", result.C, "y", "x"),
        ("D", result.D, "y", "u"),
    ):
        for table in _matrix_tables(name, matrix, rows, columns, console.width):
            console.print(table)
    console.print()
