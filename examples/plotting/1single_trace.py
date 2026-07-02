"""
Single Trace Plot
=================

This example plots one pressure history from water_hammer.h5.

The dark theme is the default FullPlot theme, so theme="dark" is optional.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"

file = fplt.open(filename)
run = file.at("/Water_Hammer/transient/runs/base")

run.plot(
    x="time",
    y="components/Pipe_Node_5/pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Pipe Node 5 Pressure",
    save=example_dir / "1single_trace.png",
)
