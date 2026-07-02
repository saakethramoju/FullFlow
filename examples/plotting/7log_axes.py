"""
Logarithmic Axes
================

This example demonstrates xscale, yscale, and y2scale.

Log scales only change the plotted axis. They do not transform the stored data.
All values plotted on a log axis must be positive.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
log_data = file.at("/log_data")

# Bode-style plot with a logarithmic frequency axis.
log_data.plot(
    x="frequency",
    y="gain",
    y2="phase_lag",
    labels="Gain",
    y2labels="Phase Lag",
    xlabel="Frequency [Hz]",
    ylabel="Gain [-]",
    y2label="Phase Lag [deg]",
    title="Frequency Response with a Log X-Axis",
    xscale="log",
    save=example_dir / "7log_x_axis.png",
    show=False,
)

# Decaying residual-style data on a logarithmic y-axis.
log_data.plot(
    x="time",
    y=[
        "positive_decay",
        "positive_growth",
    ],
    labels=[
        "Positive Decay",
        "Positive Growth",
    ],
    xlabel="Time [s]",
    ylabel="Value [-]",
    title="Positive Data on a Log Y-Axis",
    yscale="log",
    save=example_dir / "7log_y_axis.png",
    show=False,
)

fplt.show()
