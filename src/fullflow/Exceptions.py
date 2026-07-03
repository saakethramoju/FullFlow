"""
FullFlow-specific exception types.

These exceptions keep user-facing errors clear without creating a different
exception class for every component or every possible invalid value.

These classes are intended for high-level FullFlow categories
such as unassigned States, map loading, solver setup, and solver
convergence.
"""


class FullFlowError(Exception):
    """Base class for all FullFlow-specific errors."""


class FullFlowConfigurationError(FullFlowError, ValueError):
    """A model, component, balance, dynamic equation, or solver setup is invalid."""


class FullFlowStateError(FullFlowError, ValueError):
    """A State is missing, invalid, derived incorrectly, or outside allowed bounds."""


class UnassignedStateError(FullFlowStateError):
    """A State was read before it had a value."""


class FullFlowMapError(FullFlowError):
    """Base class for map loading and interpolation errors."""


class MapLoadError(FullFlowMapError, ValueError):
    """An HDF5 map file or group could not be loaded correctly."""


class MapRangeError(FullFlowMapError, ValueError):
    """A map input was outside the tabulated range."""


class FullFlowSolverError(FullFlowError, RuntimeError):
    """Base class for solver failures."""


class SolverSetupError(FullFlowSolverError):
    """The nonlinear system cannot be assembled or is underdetermined."""


class SolverConvergenceError(FullFlowSolverError):
    """A steady-state solve failed or converged to unacceptable residuals."""


class TransientStepError(SolverConvergenceError):
    """A transient timestep failed after the nonlinear solve or timestep retries."""


class SensorDataStop(FullFlowSolverError):
    """A sensor requested a clean solve stop because test data is unavailable."""


__all__ = [
    "FullFlowError",
    "FullFlowConfigurationError",
    "FullFlowStateError",
    "UnassignedStateError",
    "FullFlowMapError",
    "MapLoadError",
    "MapRangeError",
    "FullFlowSolverError",
    "SolverSetupError",
    "SolverConvergenceError",
    "TransientStepError",
    "SensorDataStop",
]
