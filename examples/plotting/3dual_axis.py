"""
Dual-Axis Plot
==============

This example uses the left y-axis for valve area and the right y-axis for
outlet pressure.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"

file = fplt.open(filename)
run = file.at("/Water_Hammer/transient/runs/base")

run.plot(
    x="time",
    y="components/Outlet_Valve/cross_sectional_area",
    y2="components/Pipe_Node_5/pressure",
    labels="Valve Area",
    y2labels="Pipe Node 5 Pressure",
    xlabel="Time [s]",
    ylabel="Valve Area [m²]",
    y2label="Pressure [Pa]",
    title="Valve Closure and Downstream Pressure Response",
    save=example_dir / "3dual_axis.png",
)
