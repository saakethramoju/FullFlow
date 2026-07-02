"""
Multidimensional Dataset as Line Traces
=======================================

This example plots a 2D HDF5 dataset as multiple line traces.

The dataset has shape:

    pressure_map[station, time]

axis=-1 means the last dimension, time, is the plotted x direction. The
remaining station dimension becomes separate line traces.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
maps = file.at("/maps")

maps.plot(
    x="time",
    y="pressure_map",
    axis=-1,
    labels=[
        "Station 0.0",
        "Station 0.2",
        "Station 0.4",
        "Station 0.6",
        "Station 0.8",
        "Station 1.0",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="2D Pressure Dataset as Line Traces",
    save=example_dir / "9multidimensional_line_traces.png",
    show=False,
)

fplt.show()
