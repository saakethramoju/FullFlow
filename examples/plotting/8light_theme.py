"""
Light Theme
===========

FullPlot supports two themes: dark and light.

The dark theme is the default. This example uses the light theme explicitly.
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
    title="Pipe Node 5 Pressure, Light Theme",
    theme="light",
    save=example_dir / "8light_theme.png",
)
