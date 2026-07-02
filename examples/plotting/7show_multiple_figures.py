"""
Show Multiple Figures at the Same Time
======================================

Each plot uses show=False so the script does not stop after every figure.
The final fplt.show() call displays all currently open figures together.
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
    save=example_dir / "7node_pressures.png",
    show=False,
)

run.plot(
    x="time",
    y=[
        "components/Pipe_Segment_1/mass_flow",
        "components/Pipe_Segment_2/mass_flow",
        "components/Pipe_Segment_3/mass_flow",
        "components/Pipe_Segment_4/mass_flow",
        "components/Pipe_Segment_5/mass_flow",
    ],
    labels=[
        "Pipe Segment 1",
        "Pipe Segment 2",
        "Pipe Segment 3",
        "Pipe Segment 4",
        "Pipe Segment 5",
    ],
    xlabel="Time [s]",
    ylabel="Mass Flow [kg/s]",
    title="Pipe Segment Mass Flow",
    save=example_dir / "7segment_mass_flow.png",
    show=False,
)

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
    save=example_dir / "7pressure_heatmap.png",
    show=False,
)

fplt.show()
