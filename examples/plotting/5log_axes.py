"""
Log-Axis Plot
=============

This example plots nonlinear solver residuals on a logarithmic y-axis.

Use a log axis when the raw data is positive and spans several orders of
magnitude. If the data is already stored as log10(value), keep the axis linear.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"

file = fplt.open(filename)
run = file.at("/Water_Hammer/transient/runs/base")

run.plot(
    x="diagnostics/time",
    y=[
        "diagnostics/max_abs_residual",
        "diagnostics/rms_residual",
    ],
    labels=[
        "Max Absolute Residual",
        "RMS Residual",
    ],
    xlabel="Time [s]",
    ylabel="Residual [-]",
    yscale="log",
    title="Transient Solver Residuals",
    save=example_dir / "5solver_residuals_log.png",
)
