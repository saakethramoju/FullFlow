"""
Multidimensional Plotting
=========================

This example creates a small derived HDF5 file with one true 2D pressure array:

    pressure[node, time]

Then FullPlot plots that same 2D dataset two ways:

    - as multiple line traces using plot(..., axis=-1)
    - as a heat map using map(...)

This is useful when your HDF5 file stores map data, test matrices, or simulation
results as multidimensional arrays.
"""

from pathlib import Path

import h5py
import numpy as np

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
source_filename = example_dir / "water_hammer.h5"
derived_filename = example_dir / "water_hammer_pressure_map.h5"

source = fplt.open(source_filename)
run = source.at("/Water_Hammer/transient/runs/base")

# Read the separate pressure histories.
time = run.read("time")
pressure_map = np.vstack([
    run.read("components/Pipe_Node_1/pressure"),
    run.read("components/Pipe_Node_2/pressure"),
    run.read("components/Pipe_Node_3/pressure"),
    run.read("components/Pipe_Node_4/pressure"),
    run.read("components/Pipe_Node_5/pressure"),
])

# Store the derived 2D data in a simple HDF5 file.
with h5py.File(derived_filename, "w") as h5:
    h5["time"] = time
    h5["pipe_node"] = np.array([1, 2, 3, 4, 5])
    h5["pressure"] = pressure_map

file = fplt.open(derived_filename)

# Plot each row of pressure[node, time] as a separate line.
file.plot(
    x="time",
    y="pressure",
    axis=-1,
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Pressure Map Plotted as Line Traces",
    save=example_dir / "6pressure_map_lines.png",
    show=False,
)

# Plot the same 2D dataset as a heat map.
file.map(
    z="pressure",
    x="time",
    y="pipe_node",
    xlabel="Time [s]",
    ylabel="Pipe Node",
    zlabel="Pressure [Pa]",
    title="Pressure Map Plotted as a Heat Map",
    cmap="plasma",
    save=example_dir / "6pressure_map_heatmap.png",
    show=False,
)

fplt.show()
