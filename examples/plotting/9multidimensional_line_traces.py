"""
2D Dataset as Multiple Line Traces
==================================

This example plots one 2D HDF5 dataset as several 1D line traces.

The HDF5 group is:

    /maps

The dataset is:

    pressure_map[station, time]

with shape:

    (6, 501)

This means:

    axis 0 = station, length 6
    axis 1 = time,    length 501

The plot call uses:

    y="pressure_map"
    x="time"
    axis=-1

The key argument is axis=-1.

In Python and NumPy, axis=-1 means "the last axis". For pressure_map, the last
axis is time. Therefore FullPlot plots along the time direction.

The remaining axis, station, becomes separate traces. So the result is:

    Station 0.0 pressure versus time
    Station 0.2 pressure versus time
    Station 0.4 pressure versus time
    Station 0.6 pressure versus time
    Station 0.8 pressure versus time
    Station 1.0 pressure versus time

This is the line-plot version of the same 2D data used in the heat-map example.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
maps = file.at("/maps")


# Read the dataset only to print its shape for clarity.
# The plot call below would work without these two lines.
pressure_map = maps.read("pressure_map")
print("pressure_map shape:", pressure_map.shape, "= [station, time]")
print("axis=-1 selects the last axis, which is time.")


# Since axis=-1 selects the time axis, the x dataset must also be time.
# The station axis is not used as x. Instead, each station becomes one trace.
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
