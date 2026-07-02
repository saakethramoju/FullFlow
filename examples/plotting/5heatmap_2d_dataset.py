"""
Heat Map from a 2D Dataset
==========================

This example plots a true 2D HDF5 dataset as a heat map.

The dataset has shape:

    pressure_map[station, time]

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
maps = file.at("/maps")

maps.map(
    z="pressure_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="Pressure Map from a 2D Dataset",
    cmap="plasma",
    save=example_dir / "5heatmap_2d_dataset.png",
    show=False,
)

fplt.show()
