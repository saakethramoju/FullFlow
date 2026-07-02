"""
Multiple Traces on One Axis
===========================

This example plots several pressure histories on the same y-axis.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")

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
    title="Multiple Pressure Traces",
    save=example_dir / "3multiple_traces.png",
    show=False,
)

fplt.show()
