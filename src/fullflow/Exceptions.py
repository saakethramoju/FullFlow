"""
FullFlow-specific exception types.

These exceptions keep user-facing errors clear without creating a different
exception class for every component or every possible invalid value.

These classes are intended for high-level FullFlow categories
such as unassigned States, map loading, solver setup, and solver
convergence.
"""


class FullFlowError(Exception):
    """Base exception for all intentional FullFlow user-facing failures.

        Catch this class when application code wants to handle any configuration,
        state, map, sensor, or solver failure raised by FullFlow without catching
        unrelated Python exceptions."""


class FullFlowConfigurationError(FullFlowError, ValueError):
    """Raised when a network, component, balance, or solver setup is structurally invalid.

        Typical causes include missing required states, invalid argument
        combinations, impossible model-option selections, or residual systems that
        cannot be assembled."""


class FullFlowStateError(FullFlowError, ValueError):
    """Raised when a ``State`` or state-like object is unavailable or invalid.

        This category covers unassigned values, invalid bounds, failed numeric
        conversion, and derived-state problems that should be shown clearly to the
        model author."""


class UnassignedStateError(FullFlowStateError):
    """Raised when code reads a ``State`` value before it has been assigned.

        Use this error to distinguish a missing output or missing initial condition
        from a numerical failure in the solver."""


class FullFlowMapError(FullFlowError):
    """Base class for map-loading, map-validation, and map-interpolation errors.

        Catch this class around workflows that load HDF5 maps or interpolate
        tabulated engineering data."""


class MapLoadError(FullFlowMapError, ValueError):
    """Raised when a map file, HDF5 group, axis, output, or metadata block cannot be loaded.

        The message is intended to identify the map group and the missing or invalid
        dataset so the table can be repaired quickly."""


class MapRangeError(FullFlowMapError, ValueError):
    """Raised when map extrapolation is disabled and an input leaves the tabulated domain.

        Either widen the map, change the model initial condition, or construct the
        map with ``extrapolate=True`` if extrapolation is physically acceptable."""


class FullFlowSolverError(FullFlowError, RuntimeError):
    """Base class for intentional steady-state and transient solver failures.

        This class covers setup, convergence, timestep, and clean-stop conditions
        produced by the solver layer."""


class SolverSetupError(FullFlowSolverError):
    """Raised when a nonlinear residual system cannot be assembled or evaluated.

        Typical causes include inconsistent balance counts, invalid dynamic-equation
        tuples, missing iteration variables, or component evaluation errors that
        occur before a valid residual has been collected."""


class SolverConvergenceError(FullFlowSolverError):
    """Raised when SciPy terminates but FullFlow's residual acceptance check fails.

        The attached message usually includes maximum residual, RMS residual, and
        variable/residual labels to help identify the failed equation."""


class TransientStepError(SolverConvergenceError):
    """Raised when a transient timestep cannot be accepted after all retry attempts.

        The timestep is rolled back before this error is raised so the network state
        remains at the last accepted time."""


class SensorDataStop(FullFlowSolverError):
    """Internal clean-stop signal raised when a sensor with ``extend=False`` runs out of data.

        The transient solver converts this into a normal stop reason instead of
        treating it as a numerical failure."""


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
