"""
Module-Level API
================

Most examples use:

    file = fplt.open(filename)
    run = file.at("/some/group")
    run.plot(...)

FullPlot also provides module-level helpers for quick one-off plots.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

# Print a short tree from the file root.
fplt.tree(filename, max_depth=2)

# Plot directly from a group using the root argument.
fplt.plot(
    filename,
    root="/demo_transient",
    x="time",
    y="mass_flow",
    xlabel="Time [s]",
    ylabel="Mass Flow [kg/s]",
    title="Module-Level Plot Call",
    save=example_dir / "13module_level_plot.png",
    show=False,
)

# Make a heat map directly from a group using the root argument.
fplt.map(
    filename,
    root="/maps",
    z="temperature_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Temperature [K]",
    title="Module-Level Map Call",
    cmap="inferno",
    save=example_dir / "13module_level_map.png",
    show=False,
)

fplt.show()
