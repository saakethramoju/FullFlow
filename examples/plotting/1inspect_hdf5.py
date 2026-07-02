"""
Inspect an HDF5 File
====================

This is the first thing to do with a new HDF5 result file.

FullPlot does not require you to know the entire HDF5 layout ahead of time. Use
this example to see what groups and datasets are available before making plots.

Run 0generate_plotting_data.py first if plotting_demo.h5 does not exist.
"""

from pathlib import Path

from fullflow import fullplot as fplt


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"


# Open the file from disk.
# This does not load the whole HDF5 file into memory. It just creates a small
# FullPlot file object that knows the filename and current root path.
file = fplt.open(filename)


# ---------------------------------------------------------------------------
# Print the HDF5 tree
# ---------------------------------------------------------------------------
# tree() shows groups and datasets under the current root.
# max_depth=3 keeps the output short.
# ---------------------------------------------------------------------------

file.tree(max_depth=3)

print()


# ---------------------------------------------------------------------------
# List datasets by shape
# ---------------------------------------------------------------------------
# list() groups datasets into 1D, 2D, 3D+, scalar, and non-numeric datasets.
# This helps you decide whether to use plot(), map(), or read().
# ---------------------------------------------------------------------------

file.list()

print()


# ---------------------------------------------------------------------------
# Scope into a group
# ---------------------------------------------------------------------------
# file.at("/demo_transient") returns a new object whose root is the
# /demo_transient group. After that, names such as "time" and "node_pressure"
# are resolved relative to /demo_transient.
# ---------------------------------------------------------------------------

run = file.at("/demo_transient")
run.list()

print()


# ---------------------------------------------------------------------------
# Read scalar values
# ---------------------------------------------------------------------------
# values() is for scalar datasets, such as constants or summary values.
# ---------------------------------------------------------------------------

file.values("/scalars")

print()


# ---------------------------------------------------------------------------
# Read numeric arrays directly
# ---------------------------------------------------------------------------
# read() returns a NumPy array. Use this when you want to inspect values or do a
# calculation before plotting.
# ---------------------------------------------------------------------------

time = run.read("time")
pressure = run.read("node_pressure")

print("Number of time points:", len(time))
print("Initial node pressure [Pa]:", pressure[0])
print("Final node pressure [Pa]:", pressure[-1])
