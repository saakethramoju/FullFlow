"""
Single Trace Plot
=================

This example plots one 1D dataset against another 1D dataset.

The HDF5 group is:

    /demo_transient

The datasets are:

    time[node]             1D x-axis array
    node_pressure[time]    1D y-axis array

The plot call means:

    x-axis = time
    y-axis = node_pressure

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")


# Plot a single pressure time history.
#
# x="time" tells FullPlot to read /demo_transient/time.
# y="node_pressure" tells FullPlot to read /demo_transient/node_pressure.
# xlabel, ylabel, and title are normal plot labels.
# save writes the figure to a PNG file.
# show=False delays display until fplt.show() at the end.
run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Node Pressure Time History",
    save=example_dir / "2single_trace.png",
    show=False,
)

fplt.show()
