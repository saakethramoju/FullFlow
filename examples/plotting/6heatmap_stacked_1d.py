"""
Heat Map from Separate 1D Datasets
==================================

This example makes a heat map from several separate 1D datasets.

Some HDF5 files store data like this:

    station_1_pressure[time]
    station_2_pressure[time]
    station_3_pressure[time]
    ...

instead of storing one 2D array like this:

    pressure_map[station, time]

FullPlot can still make a heat map. When z is a list of 1D datasets, FullPlot
stacks them internally in the order provided:

    z=[station_1_pressure,
       station_2_pressure,
       station_3_pressure,
       ...]

becomes:

    stacked_pressure[station, time]

This example is useful for exported solver data where each sensor, node, or
station is saved as its own time history.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
traces = file.at("/separate_traces")


# Each dataset in z is 1D and has the same length as time.
# FullPlot stacks them into a 2D array with six rows.
# y="station" supplies the y-coordinate for those six rows.
traces.map(
    z=[
        "station_1_pressure",
        "station_2_pressure",
        "station_3_pressure",
        "station_4_pressure",
        "station_5_pressure",
        "station_6_pressure",
    ],
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="Pressure Heat Map from Separate 1D Datasets",
    cmap="plasma",
    save=example_dir / "6heatmap_stacked_1d.png",
    show=False,
)

fplt.show()
