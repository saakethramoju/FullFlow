"""
Inspect an HDF5 File
====================

This example shows the first thing to do with a new HDF5 file:

    - print the file tree
    - scope into a group
    - list numeric datasets
    - read scalar values
    - read a dataset directly

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"

file = fplt.open(filename)

# Print the top-level file tree.
file.tree(max_depth=3)

print()

# Print numeric datasets by shape.
file.list()

print()

# Scope to a group inside the file.
run = file.at("/demo_transient")
run.list()

print()

# Read scalar values from a group.
file.values("/scalars")

print()

# Read one dataset directly as a NumPy array.
time = run.read("time")
pressure = run.read("node_pressure")

print("Number of time points:", len(time))
print("Initial node pressure [Pa]:", pressure[0])
print("Final node pressure [Pa]:", pressure[-1])
