"""Modular steady-state solver implementation.

The package is intentionally divided by responsibility:

``solver``
    User-facing :class:`SteadyState` wrapper.
``runtime``
    Fast cached view of a network during solving.
``evaluation``
    Fixed-point passes for component-derived states.
``operations``
    One-shot static evaluation and SciPy least-squares solves.
``models``
    Model-option selection, fallback, and failure reporting helpers.
``results``
    Result formatting and model-option file export helpers.
``diagnostics``
    Rich terminal tables for verbose output.
"""

from .solver import SteadyState

__all__ = ["SteadyState"]
