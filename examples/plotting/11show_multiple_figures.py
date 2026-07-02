"""
Show Multiple Figures Together
==============================

This example creates several figures and displays them at the end.

Each plot call uses:

    show=False

That prevents each figure from blocking the script immediately. At the end,
this line displays all open figures together:

    fplt.show()

This is useful when a script creates several plots from the same HDF5 file.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")
maps = file.at("/maps")


# First figure: one pressure time history.
run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Node Pressure",
    show=False,
)


# Second figure: one mass-flow time history.
run.plot(
    x="time",
    y="mass_flow",
    xlabel="Time [s]",
    ylabel="Mass Flow [kg/s]",
    title="Mass Flow",
    show=False,
)


# Third figure: a 2D temperature map.
maps.map(
    z="temperature_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Temperature [K]",
    title="Temperature Heat Map",
    show=False,
)


# Display all three figures at the same time.
fplt.show()
