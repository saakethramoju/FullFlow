"""
Slicing a 3D Dataset
====================

This example plots slices from a true 3D HDF5 dataset.

The dataset has shape:

    pressure_3d[case, station, time]

The slice argument reduces the 3D array before plotting:

    slice={0: 1}

means "select case index 1".

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
multidim = file.at("/multidimensional")

# Slice the 3D dataset down to pressure_3d[station, time], then plot as lines.
multidim.plot(
    x="time",
    y="pressure_3d",
    axis=-1,
    slice={0: 1},
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
    title="3D Pressure Dataset, Case 1 as Line Traces",
    save=example_dir / "10pressure_3d_case1_lines.png",
    show=False,
)

# Slice the same 3D dataset down to pressure_3d[station, time], then plot as a map.
multidim.map(
    z="pressure_3d",
    x="time",
    y="station",
    slice={0: 1},
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="3D Pressure Dataset, Case 1 as a Heat Map",
    cmap="plasma",
    save=example_dir / "10pressure_3d_case1_heatmap.png",
    show=False,
)

fplt.show()
