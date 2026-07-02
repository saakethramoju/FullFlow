"""
Saving Figures
==============

The save argument can write any normal Matplotlib-supported figure format.
Common choices are .png, .pdf, and .svg.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")

run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Saved as PNG",
    save=example_dir / "14saved_plot.png",
    show=False,
)

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
