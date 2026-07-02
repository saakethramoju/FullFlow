"""
Dual-Axis Plot
==============

This example uses y for the left axis and y2 for the right axis.

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
    y="node_pressure",
    y2="mass_flow",
    labels="Node Pressure",
    y2labels="Mass Flow",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    y2label="Mass Flow [kg/s]",
    title="Pressure and Mass Flow on Separate Axes",
    save=example_dir / "4dual_axis.png",
    show=False,
)

fplt.show()
