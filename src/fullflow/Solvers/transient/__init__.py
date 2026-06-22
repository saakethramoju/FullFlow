"""Transient solver public API.

The transient solver is intentionally exposed through one small entry point:

    from fullflow import Transient

Components do not write timestep residuals themselves.  Components expose
``transient_variables`` and ``transient_derivatives``; the solver converts those
pairs into implicit backward-Euler residuals internally.

Schedule helpers are exported here for users who want to experiment with
open-loop time inputs before the full schedule-breakpoint phase is added.
"""

from .solver import Transient
from .schedules import Constant, Function, Ramp, Schedule, Step

__all__ = [
    "Transient",
    "Schedule",
    "Constant",
    "Step",
    "Ramp",
    "Function",
]
