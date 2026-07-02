"""
FullFlow: fluid, thermal, and propulsion network simulation.
"""

from fullflow.System import *
from fullflow.Solvers import *
from fullflow.Exports import *
from fullflow.Exceptions import *


def main() -> None:
    """Run the FullFlow command-line entry point."""
    print("FullFlow")


def __getattr__(name):
    """Lazily expose optional submodules without slowing normal imports."""

    if name == "fullplot":
        from fullflow.Plotting import fullplot
        return fullplot

    raise AttributeError(f"module 'fullflow' has no attribute {name!r}")

