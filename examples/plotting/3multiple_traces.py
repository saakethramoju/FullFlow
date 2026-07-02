"""
Multiple Traces on One Axis
===========================

This example plots several 1D datasets on the same left y-axis.

All y datasets must have the same length as the x dataset:

    time             shape (501,)
    source_pressure  shape (501,)
    node_pressure    shape (501,)
    outlet_pressure  shape (501,)

The plot call means:

    x-axis = time
    y-axis = pressure
    traces = source_pressure, node_pressure, outlet_pressure

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")


# Pass a list to y when several datasets should share the same y-axis.
# labels gives the legend names in the same order as the y list.
run.plot(
    x="time",
    y=[
        "source_pressure",
        "node_pressure",
        "outlet_pressure",
    ],
    labels=[
        "Source Pressure",
        "Node Pressure",
        "Outlet Pressure",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Multiple Pressure Traces on One Axis",
    save=example_dir / "3multiple_traces.png",
    show=False,
)

fplt.show()
