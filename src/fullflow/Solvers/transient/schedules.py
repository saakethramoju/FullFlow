"""Open-loop schedule helpers for transient inputs.

These classes are intentionally simple callables.  A ``Schedule`` component
writes ``target.value = function(network.time.value)`` during normal component
evaluation.  The transient runtime treats schedule components as prescribed
inputs, not residual equations.

Breakpoint snapping is not implemented in this phase.  The preset functions
still expose ``breakpoints`` so the future timestep picker can avoid stepping
across known discontinuities.
"""

from __future__ import annotations

from typing import Any, Callable

from fullflow.System import Component, State


class Constant:
    """Callable schedule returning one fixed value for all time."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, time: float) -> Any:
        return self.value

    @property
    def breakpoints(self) -> list[float]:
        return []


class Step:
    """Callable schedule with one instantaneous time step.

    Parameters
    ----------
    time : float
        Time at which the output switches from ``before`` to ``after``.
    before, after : Any
        Values returned before and after the step.
    """

    def __init__(self, time: float, before: Any, after: Any) -> None:
        self.time = float(time)
        self.before = before
        self.after = after

    def __call__(self, time: float) -> Any:
        return self.before if time < self.time else self.after

    @property
    def breakpoints(self) -> list[float]:
        return [self.time]


class Ramp:
    """Callable linear ramp between two values over a time interval."""

    def __init__(
        self,
        start_time: float,
        stop_time: float,
        start_value: float,
        stop_value: float,
    ) -> None:
        self.start_time = float(start_time)
        self.stop_time = float(stop_time)
        self.start_value = float(start_value)
        self.stop_value = float(stop_value)

        if self.stop_time <= self.start_time:
            raise ValueError("stop_time must be greater than start_time.")

    def __call__(self, time: float) -> float:
        if time <= self.start_time:
            return self.start_value
        if time >= self.stop_time:
            return self.stop_value

        fraction = (time - self.start_time) / (self.stop_time - self.start_time)
        return self.start_value + fraction * (self.stop_value - self.start_value)

    @property
    def breakpoints(self) -> list[float]:
        return [self.start_time, self.stop_time]


class Function:
    """Wrap an arbitrary user callable as a schedule function.

    ``breakpoints`` are optional metadata for future timestep edge snapping.
    FullFlow cannot discover hidden discontinuities inside arbitrary Python
    functions, so discontinuous functions should provide these explicitly.
    """

    def __init__(
        self,
        function: Callable[[float], Any],
        breakpoints: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable.")
        self.function = function
        self._breakpoints = [] if breakpoints is None else [float(t) for t in breakpoints]

    def __call__(self, time: float) -> Any:
        return self.function(time)

    @property
    def breakpoints(self) -> list[float]:
        return list(self._breakpoints)


class Schedule(Component):
    """Component that prescribes a target State from network time.

    Schedules are open-loop inputs.  They do not add residuals or iteration
    variables.  The transient runtime checks that ``target`` is not also a
    transient variable, component iteration variable, or Balance variable.
    """

    _is_fullflow_schedule = True

    def __init__(
        self,
        name: str,
        network,
        target: State,
        function: Callable[[float], Any],
    ):
        self.setup()

    def evaluate_states(self) -> None:
        # Component.setup wraps ordinary inputs in State objects.  ``target`` is
        # expected to already be a State; ``function`` may be a schedule object
        # or arbitrary callable stored inside a State wrapper.
        function = self.function.value
        self.target.value = function(self.network.time.value)

    @property
    def breakpoints(self) -> list[float]:
        function = self.function.value
        return list(getattr(function, "breakpoints", []))
