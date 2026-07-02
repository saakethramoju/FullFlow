"""
Module-Level API
================

Most examples use the object API:

    file = fplt.open(filename)
    run = file.at(group)
    run.plot(...)

For quick scripts, FullPlot also has module-level helpers. Use root=... to scope
where the plot should look for datasets.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"
root = "/Water_Hammer/transient/runs/base"

fplt.plot(
    filename,
    root=root,
    x="time",
    y="components/Pipe_Node_5/pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Pipe Node 5 Pressure using fplt.plot",
    save=example_dir / "9module_level_api.png",
)
