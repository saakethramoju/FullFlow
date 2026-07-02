"""
Slicing a 3D Dataset
====================

This example plots slices from a true 3D HDF5 dataset.

The HDF5 group is:

    /multidimensional

The dataset is:

    pressure_3d[case, station, time]

with shape:

    (3, 6, 501)

This means:

    axis 0 = case,    length 3
    axis 1 = station, length 6
    axis 2 = time,    length 501

The examples below use:

    slice={0: 1}

This means:

    before plotting, select index 1 from axis 0

Since axis 0 is case, slice={0: 1} selects case 1.

Before slicing:

    pressure_3d[case, station, time]

After slice={0: 1}:

    pressure_3d[station, time]

After slicing, the data is 2D, so it can be plotted either as multiple line
traces with plot() or as a heat map with map().

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
multidim = file.at("/multidimensional")


# Print the shape to make the slicing explicit.
# The plot calls below would work without these lines.
pressure_3d = multidim.read("pressure_3d")
print("pressure_3d shape:", pressure_3d.shape, "= [case, station, time]")
print("slice={0: 1} selects case index 1 and leaves [station, time].")
print("axis=-1 then selects the time axis for line plotting.")


# ---------------------------------------------------------------------------
# 3D dataset sliced into line traces
# ---------------------------------------------------------------------------
# Step 1:
#     pressure_3d[case, station, time]
#
# Step 2:
#     slice={0: 1} selects case 1.
#
# Step 3:
#     The remaining data is pressure_3d[station, time].
#
# Step 4:
#     axis=-1 means plot along the last axis, which is time.
#
# Result:
#     one pressure-versus-time trace for each station.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3D dataset sliced into a heat map
# ---------------------------------------------------------------------------
# Step 1:
#     pressure_3d[case, station, time]
#
# Step 2:
#     slice={0: 1} selects case 1.
#
# Step 3:
#     The remaining data is pressure_3d[station, time].
#
# Step 4:
#     map() displays that 2D slice as:
#
#         x-axis = time
#         y-axis = station
#         color  = pressure
# ---------------------------------------------------------------------------

multidim.map(
    z="pressure_3d",
    x="time",
    y="station",
    slice={0: 1},
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="3D Pressure Dataset, Case 1 as a Heat Map",
    save=example_dir / "10pressure_3d_case1_heatmap.png",
    show=False,
)

fplt.show()
