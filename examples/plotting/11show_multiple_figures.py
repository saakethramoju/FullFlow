"""
Show Multiple Figures Together
==============================

This example creates several plots with show=False and then displays all open
figures at the end with fplt.show().

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")
maps = file.at("/maps")

run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Node Pressure",
    show=False,
)

run.plot(
    x="time",
    y="mass_flow",
    xlabel="Time [s]",
    ylabel="Mass Flow [kg/s]",
    title="Mass Flow",
    show=False,
)

maps.map(
    z="temperature_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Temperature [K]",
    title="Temperature Heat Map",
    cmap="inferno",
    show=False,
)

fplt.show()
