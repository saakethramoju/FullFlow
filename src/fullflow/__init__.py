"""
FullFlow: fluid, thermal, and propulsion network simulation.
"""

__version__ = "2.0.1"

from fullflow.System import *
from fullflow.Solvers import *
from fullflow.Exports import *
from fullflow.Exceptions import *


def main() -> None:
    """Run the FullFlow command-line entry point."""
    print("FullFlow")
