"""
Heat Map from a 2D Dataset
==========================

This example plots a true 2D HDF5 dataset as a heat map.

The HDF5 group is:

    /maps

The relevant datasets are:

    time          shape (501,)
    station       shape (6,)
    pressure_map  shape (6, 501)

The 2D pressure dataset is arranged as:

    pressure_map[station, time]

That means:

    axis 0 = station direction
    axis 1 = time direction

The map call means:

    x-axis = time
    y-axis = station
    color  = pressure_map[station, time]

A heat map is often the clearest way to view a 2D field because the third
quantity is represented by color.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
maps = file.at("/maps")


# z is the 2D value array that becomes the color field.
# x is the horizontal coordinate array.
# y is the vertical coordinate array.
# zlabel labels the colorbar.
# high values to yellow.
maps.map(
    z="pressure_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="Pressure Map from a 2D Dataset",
    save=example_dir / "5heatmap_2d_dataset.png",
    show=False,
)

fplt.show()
