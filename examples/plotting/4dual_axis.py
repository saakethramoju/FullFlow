"""
Dual-Axis Plot
==============

This example plots two quantities with different units on the same x-axis.

The left y-axis uses y:

    node_pressure [Pa]

The right y-axis uses y2:

    mass_flow [kg/s]

Use a dual-axis plot when the datasets share the same x-axis but have different
units or very different magnitudes.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")


# y is plotted on the left axis.
# y2 is plotted on the right axis.
# labels names the left-axis trace.
# y2labels names the right-axis trace.
# ylabel names the left y-axis.
# y2label names the right y-axis.
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
