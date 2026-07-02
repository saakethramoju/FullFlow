"""
Multiple Traces on One Axis
===========================

This example plots all five pipe-node pressure histories on the same axes.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"

file = fplt.open(filename)
run = file.at("/Water_Hammer/transient/runs/base")

run.plot(
    x="time",
    y=[
        "components/Pipe_Node_1/pressure",
        "components/Pipe_Node_2/pressure",
        "components/Pipe_Node_3/pressure",
        "components/Pipe_Node_4/pressure",
        "components/Pipe_Node_5/pressure",
    ],
    labels=[
        "Pipe Node 1",
        "Pipe Node 2",
        "Pipe Node 3",
        "Pipe Node 4",
        "Pipe Node 5",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Water Hammer Pressure Wave",
    save=example_dir / "2multiple_pressure_traces.png",
)
