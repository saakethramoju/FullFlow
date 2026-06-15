"""Public steady-state solver import.

The implementation lives in :mod:`fullflow.Solvers.steady_state` so the
public API remains stable while the solver internals stay modular.
"""

from .steady_state import SteadyState

__all__ = ["SteadyState"]
