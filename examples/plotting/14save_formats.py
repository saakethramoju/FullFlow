"""
Saving Figures
==============

The save argument writes a Matplotlib figure to disk.

The filename extension controls the output format:

    .png    raster image, good for quick viewing
    .svg    vector image, good for diagrams and web pages
    .pdf    vector document, good for reports

This example saves one plot as PNG and another as SVG.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")


# Save a pressure plot as a PNG file.
run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Saved as PNG",
    save=example_dir / "14saved_plot.png",
    show=False,
)


# Save a mass-flow plot as an SVG file.
run.plot(
    x="time",
    y="mass_flow",
    xlabel="Time [s]",
    ylabel="Mass Flow [kg/s]",
    title="Saved as SVG",
    save=example_dir / "14saved_plot.svg",
    show=False,
)

fplt.show()
