"""
FullFlow plotting tools.

Recommended use
---------------

from fullflow import fullplot as fplt

file = fplt.open("run.h5")
file.tree()
file.list()

run = file.at("/Pipe_Network/transient/runs/base")

run.plot(
    x="time",
    y="tracks/Pipe_Mass_Flow_[kg_s]",
)
"""

from fullflow.Plotting.fullplot import (
    FullPlotError,
    DatasetNotFoundError,
    AmbiguousDatasetError,
    PlotDataError,
    H5File,
    open,
    tree,
    list,
    read,
    values,
    plot,
    map,
    show,
)

__all__ = [
    "FullPlotError",
    "DatasetNotFoundError",
    "AmbiguousDatasetError",
    "PlotDataError",
    "H5File",
    "open",
    "tree",
    "list",
    "read",
    "values",
    "plot",
    "map",
    "show",
]