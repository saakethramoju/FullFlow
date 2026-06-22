"""
FullFlow: fluid, thermal, and propulsion network simulation.
"""

from fullflow.System import *
from fullflow.Solvers import *
from fullflow.Exports import *


from fullflow.plotting import (
    h5_arrays,
    h5_attrs,
    h5_datasets,
    h5_find,
    h5_groups,
    h5_imshow,
    h5_latest_solution,
    h5_map,
    h5_map_groups,
    h5_plot,
    h5_plot_steps,
    h5_print,
    h5_read,
    h5_solution,
    h5_solution_groups,
    h5_solution_value,
    h5_table,
    h5_table_groups,
    h5_track,
    h5_track_names,
    h5_tracks,
    h5_transient_groups,
)

def main() -> None:
    """Run the FullFlow command-line entry point."""
    print("FullFlow")
