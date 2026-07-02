"""
Light Theme
===========

FullPlot uses a dark theme by default.

This example switches to the light theme with:

    theme="light"

Only the visual style changes. The data selection is the same as any other
plot() call.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)
run = file.at("/demo_transient")


# This is the same kind of multi-trace pressure plot shown earlier, but with a
# light background instead of the default dark background.
run.plot(
    x="time",
    y=[
        "node_pressure",
        "source_pressure",
    ],
    labels=[
        "Node Pressure",
        "Source Pressure",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Light Theme Plot",
    theme="light",
    save=example_dir / "12light_theme.png",
    show=False,
)

fplt.show()
