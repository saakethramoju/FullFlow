"""
Inspect an HDF5 result file with FullPlot
========================================

This example shows the first thing to do with a new HDF5 file:

    - print the file tree
    - scope into a group
    - list numeric datasets that can be plotted
    - read one dataset directly

The file water_hammer.h5 should be in this same folder.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "water_hammer.h5"

file = fplt.open(filename)

# Print the top-level tree. max_depth keeps the output readable.
file.tree(max_depth=4)

# Scope to the transient run group.
run = file.at("/Water_Hammer/transient/runs/base")

# List the datasets under this group.
run.list()

# Read a dataset directly as a NumPy array.
time = run.read("time")
pressure = run.read("components/Pipe_Node_5/pressure")

print()
print("Number of time points:", len(time))
print("Initial Pipe Node 5 pressure [Pa]:", pressure[0])
print("Final Pipe Node 5 pressure [Pa]:", pressure[-1])
