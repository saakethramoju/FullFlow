"""
Light Theme
===========

FullPlot uses a dark theme by default. This example switches to the light theme.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")

run.plot(
    x="time",
    y=[
        "node_pressure",
        "source_pressure",
    ],
    labels=[
        "Node Pressure",
        "Source Pressure",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Light Theme Plot",
    theme="light",
    save=example_dir / "12light_theme.png",
    show=False,
)

fplt.show()
