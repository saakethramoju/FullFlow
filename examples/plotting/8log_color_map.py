"""
Logarithmic Color Map
=====================

This example demonstrates zscale="log" for a heat map.

The x and y axes remain linear, but the colorbar uses a logarithmic scale.
All z values must be positive.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
maps = file.at("/maps")

maps.map(
    z="positive_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Normalized Positive Value [-]",
    title="Heat Map with a Log Color Scale",
    zscale="log",
    cmap="viridis",
    save=example_dir / "8log_color_map.png",
    show=False,
)

fplt.show()
