"""
Heat Map from Separate 1D Datasets
==================================

This example builds a pressure heat map from five separate pressure histories.

Each selected pressure dataset is 1D. FullPlot stacks the selected 1D datasets
into one 2D array before plotting the map.
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
    title="Water Hammer Pressure Heat Map",
    cmap="plasma",
    save=example_dir / "4pressure_heatmap.png",
)
