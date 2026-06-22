"""
FullFlow: fluid, thermal, and propulsion network simulation.
"""

from fullflow.System import *
from fullflow.Solvers import *
from fullflow.Exports import *

from fullflow.plotting import (
    h5_attributes,
    h5_components,
    h5_datasets,
    h5_groups,
    h5_history,
    h5_map,
    h5_maps,
    h5_networks,
    h5_objects,
    h5_plot,
    h5_plot_steps,
    h5_print,
    h5_read,
    h5_solution,
    h5_solution_value,
    h5_table,
    h5_track,
    h5_track_names,
    h5_tracks,
    h5_transient_groups,
)


def main() -> None:
    """Run the FullFlow command-line entry point."""
    print("FullFlow")
