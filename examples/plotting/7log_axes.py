"""
Logarithmic Axes
================

This example demonstrates:

    xscale="log"
    yscale="log"

A log scale changes only the displayed axis. It does not apply log10() to the
stored data.

Important rule:

    Every value plotted on a log axis must be positive.

The generated file includes positive data specifically for this example.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
log_data = file.at("/log_data")


# ---------------------------------------------------------------------------
# Logarithmic x-axis
# ---------------------------------------------------------------------------
# frequency is log-spaced and strictly positive.
# gain is plotted on the left y-axis.
# phase_lag is plotted on the right y-axis using y2.
# xscale="log" makes the frequency axis logarithmic.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Logarithmic y-axis
# ---------------------------------------------------------------------------
# positive_decay and positive_growth are both strictly positive.
# yscale="log" makes the left y-axis logarithmic.
# ---------------------------------------------------------------------------

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
