"""
Logarithmic Color Map
=====================

This example demonstrates zscale="log" for a heat map.

For a heat map:

    xscale controls the x-axis
    yscale controls the y-axis
    zscale controls the colorbar

Here, the x and y axes remain linear, but the color scale is logarithmic.

Important rule:

    Every z value must be positive when zscale="log".

The dataset positive_map is positive by construction.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
maps = file.at("/maps")


# z="positive_map" is a 2D positive array with shape [station, time].
# zscale="log" makes the colorbar logarithmic.
maps.map(
    z="positive_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Normalized Positive Value [-]",
    title="Heat Map with a Log Color Scale",
    zscale="log",
    save=example_dir / "8log_color_map.png",
    show=False,
)

fplt.show()
