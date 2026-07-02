"""
Log Color Scale Map
===================

This example uses a logarithmic color scale with zscale="log".

All z values must be positive for a logarithmic color scale.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"

file = fplt.open(filename)
run = file.at("/Water_Hammer/transient/runs/base")

run.map(
    z=[
        "components/Pipe_Node_1/pressure",
        "components/Pipe_Node_2/pressure",
        "components/Pipe_Node_3/pressure",
        "components/Pipe_Node_4/pressure",
        "components/Pipe_Node_5/pressure",
    ],
    x="time",
    y=[1, 2, 3, 4, 5],
    xlabel="Time [s]",
    ylabel="Pipe Node",
    zlabel="Pressure [Pa]",
    zscale="log",
    title="Water Hammer Pressure Heat Map, Log Color Scale",
    cmap="plasma",
    save=example_dir / "10log_color_map.png",
)
