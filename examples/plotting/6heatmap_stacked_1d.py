"""
Heat Map from Separate 1D Datasets
==================================

This example shows how to make a heat map when the HDF5 file stores each trace
as a separate 1D dataset instead of one 2D array.

FullPlot stacks the selected 1D datasets into a 2D array internally.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
traces = file.at("/separate_traces")

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
