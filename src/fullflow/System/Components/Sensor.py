from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from fullplot import Trace

from fullflow.Exceptions import SensorDataStop, SolverSetupError
from fullflow.System import Component, State
from fullflow.System.State import is_assignable_state_like, is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


SENSOR_DATA_ROLE = "data"
SENSOR_CONDITION_ROLES = {"redline", "blueline", "yellowline", "greenline"}
SENSOR_CONDITION_ACTIONS = {
    "redline": "event",
    "yellowline": "warning",
    "blueline": "event",
    "greenline": "event",
}


@dataclass(slots=True)
class SensorEvent:
    """One runtime crossing of a sensor condition trace."""

    time: float
    sensor: str
    trace: str
    role: str
    action: str
    reading: float
    line_value: float
    previous_reading: float
    previous_line_value: float
    crossing_direction: str
    message: str

    @property
    def is_redline_abort(self) -> bool:
        # Redlines are high-severity Sensor events. They no longer stop the
        # transient solver by themselves; abort behavior should be modeled by
        # Sequence command logic if desired.
        return False

    def as_record(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "sensor": self.sensor,
            "trace": self.trace,
            "role": self.role,
            "action": self.action,
            "reading": self.reading,
            "line_value": self.line_value,
            "previous_reading": self.previous_reading,
            "previous_line_value": self.previous_line_value,
            "crossing_direction": self.crossing_direction,
            "message": self.message,
        }


@dataclass(slots=True)
class SensorCondition:
    """FullPlot line trace attached to a Sensor for transient event checks."""

    trace: Trace
    role: str
    action: str
    previous_reading: float = math.nan
    previous_line_value: float = math.nan
    was_active: bool = False

    @property
    def name(self) -> str:
        return getattr(self.trace, "name", "condition")


class Sensor(Component):
    """Virtual instrumentation channel.

    In its simplest form, a Sensor behaves like a named ``track()`` entry. It
    exports the value of ``reading`` under the sensor's name.

    If both ``variable`` and ``data`` are supplied, the Sensor also behaves like
    a Balance. The solver adjusts ``variable`` until ``reading`` matches the
    supplied FullPlot Trace value sampled at the current network time.

    ``data`` is reserved for a FullPlot Trace with role ``"data"``. Optional
    ``conditions`` are FullPlot Trace objects with role ``"redline"``,
    ``"blueline"``, ``"yellowline"``, or ``"greenline"``. Conditions are
    checked after accepted transient steps and logged as sensor events.
    Redlines are high-severity events, but they do not stop the run by
    themselves. Abort behavior should be modeled with Sequence commands.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        reading: State,
        variable: State | None = None,
        data: Trace | None = None,
        conditions: Trace | list[Trace] | tuple[Trace, ...] | None = None,
        extend: bool = True,
    ) -> None:
        if data is not None:
            self._validate_data_trace(name, data)

        condition_list = self._normalize_conditions(name, conditions)

        self.setup()

        # Component.setup wraps every constructor input in a State by default.
        # FullPlot Trace objects are external data/limit objects, not solver
        # states. Keep them as the original Trace-backed objects.
        self.data = data
        self.conditions = tuple(condition_list)

        # Export/cache states. These are updated every evaluate_states() pass
        # and during residual evaluation.
        self.data_value = State(math.nan)
        self.error = State(math.nan)
        self.active = State(False)
        self.variable_value = State(math.nan)

    @property
    def value(self):
        return self.reading.value

    @property
    def ignored_export_attributes(self) -> set[str]:
        # Keep external FullPlot Trace objects and configuration metadata out of
        # the component table. The useful sampled values are exported through
        # reading/data_value/error/active/variable_value. Condition traces and
        # sparse condition events are exported through dedicated HDF5 groups.
        return {"data", "conditions", "extend"}

    @staticmethod
    def _trace_role(trace: Trace) -> str:
        return str(getattr(trace, "role", SENSOR_DATA_ROLE) or SENSOR_DATA_ROLE).lower().strip()

    @staticmethod
    def _validate_trace_object(sensor_name: str, trace: Any, *, purpose: str) -> None:
        if not isinstance(trace, Trace):
            raise SolverSetupError(
                f"Sensor {sensor_name!r} {purpose} must be a fullplot.Trace object. "
                f"Got {type(trace).__name__}."
            )

    @classmethod
    def _validate_data_trace(cls, sensor_name: str, trace: Trace) -> None:
        cls._validate_trace_object(sensor_name, trace, purpose="data")
        role = cls._trace_role(trace)
        if role != SENSOR_DATA_ROLE:
            raise SolverSetupError(
                f"Sensor {sensor_name!r} data must have role='data'. "
                f"Trace {getattr(trace, 'name', '<unnamed>')!r} has role={role!r}."
            )

    @classmethod
    def _normalize_conditions(
        cls,
        sensor_name: str,
        conditions: Trace | list[Trace] | tuple[Trace, ...] | None,
    ) -> list[SensorCondition]:
        if conditions is None:
            return []

        if isinstance(conditions, Trace):
            traces = [conditions]
        else:
            try:
                traces = list(conditions)
            except TypeError as exc:
                raise SolverSetupError(
                    f"Sensor {sensor_name!r} conditions must be a fullplot.Trace object "
                    "or a list/tuple of fullplot.Trace objects."
                ) from exc

        normalized: list[SensorCondition] = []
        for trace in traces:
            cls._validate_trace_object(sensor_name, trace, purpose="condition")
            role = cls._trace_role(trace)
            trace_name = getattr(trace, "name", "<unnamed>")

            if role == "command":
                raise SolverSetupError(
                    f"Sensor {sensor_name!r} cannot consume command trace {trace_name!r}. "
                    "Command traces should drive component inputs or sequences, not sensor conditions."
                )

            if role not in SENSOR_CONDITION_ROLES:
                valid = ", ".join(repr(item) for item in sorted(SENSOR_CONDITION_ROLES))
                raise SolverSetupError(
                    f"Sensor {sensor_name!r} conditions must have role in {{{valid}}}. "
                    f"Trace {trace_name!r} has role={role!r}."
                )

            normalized.append(
                SensorCondition(
                    trace=trace,
                    role=role,
                    action=SENSOR_CONDITION_ACTIONS[role],
                )
            )

        return normalized

    @staticmethod
    def _resolved(value: Any, default: Any = None) -> Any:
        if is_state_like(value):
            try:
                return value.value
            except Exception:
                return default
        return value

    @property
    def _extend_enabled(self) -> bool:
        return bool(self._resolved(self.extend, default=True))

    @property
    def has_data(self) -> bool:
        return self.data is not None

    @property
    def has_variable(self) -> bool:
        return is_assignable_state_like(self.variable) and self.variable.is_assigned

    @property
    def is_anchor(self) -> bool:
        return self.has_data and self.has_variable

    @property
    def has_conditions(self) -> bool:
        return bool(self.conditions)

    def condition(self, name: str) -> SensorCondition | None:
        """Return a named condition attached to this Sensor, if present."""
        name = str(name)
        for condition in self.conditions:
            if str(condition.name) == name:
                return condition
        return None

    def has_condition(self, name: str) -> bool:
        """Return True when this Sensor owns a condition with this name."""
        return self.condition(name) is not None

    @staticmethod
    def _finite_number(value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return math.nan
        return number if math.isfinite(number) else math.nan

    @staticmethod
    def _trace_arrays_for(trace: Trace, sensor_name: str, purpose: str) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(trace, Trace):
            raise SolverSetupError(
                f"Sensor {sensor_name!r} {purpose} must be a fullplot.Trace object. "
                f"Got {type(trace).__name__}."
            )

        x_array = np.asarray(trace.x, dtype=float)
        y_array = np.asarray(trace.y, dtype=float)

        if x_array.ndim != 1 or y_array.ndim != 1:
            raise SolverSetupError(
                f"Sensor {sensor_name!r} FullPlot Trace {getattr(trace, 'name', '<unnamed>')!r} "
                "x and y arrays must be one-dimensional."
            )

        if x_array.shape != y_array.shape:
            raise SolverSetupError(
                f"Sensor {sensor_name!r} FullPlot Trace {getattr(trace, 'name', '<unnamed>')!r} "
                f"x and y arrays must have the same length. Got {len(x_array)} and {len(y_array)}."
            )

        if len(x_array) == 0:
            raise SolverSetupError(
                f"Sensor {sensor_name!r} FullPlot Trace {getattr(trace, 'name', '<unnamed>')!r} is empty."
            )

        if not np.all(np.isfinite(x_array)):
            raise SolverSetupError(
                f"Sensor {sensor_name!r} FullPlot Trace {getattr(trace, 'name', '<unnamed>')!r} "
                "time values must be finite."
            )

        if len(x_array) > 1 and np.any(np.diff(x_array) <= 0.0):
            raise SolverSetupError(
                f"Sensor {sensor_name!r} FullPlot Trace {getattr(trace, 'name', '<unnamed>')!r} "
                "time values must be strictly increasing."
            )

        return x_array, y_array

    def _trace_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        return self._trace_arrays_for(self.data, self.name, "data")

    def _sample_trace_previous(self, time_value: float) -> float:
        """Sample the FullPlot data Trace using the previous data point.

        FullFlow solver time takes priority over the test-data sample rate. At
        each solver time, the sensor uses the most recent trace sample whose
        shifted FullPlot time is less than or equal to the current solver time.
        NaN y-values, out-of-range times, and invalid times return NaN.
        """
        if self.data is None or not math.isfinite(time_value):
            return math.nan

        x, y = self._trace_arrays()

        if time_value < x[0] or time_value > x[-1]:
            return math.nan

        index = int(np.searchsorted(x, time_value, side="right") - 1)
        if index < 0 or index >= len(y):
            return math.nan

        return self._finite_number(y[index])

    def _sample_condition_trace(self, condition: SensorCondition, time_value: float) -> float:
        if not math.isfinite(time_value):
            return math.nan

        try:
            return self._finite_number(condition.trace(time_value))
        except Exception:
            # Fall back to the same previous-sample implementation used by data
            # traces so older FullPlot Trace objects remain compatible.
            x, y = self._trace_arrays_for(condition.trace, self.name, "condition")
            if time_value < x[0] or time_value > x[-1]:
                return math.nan
            index = int(np.searchsorted(x, time_value, side="right") - 1)
            if index < 0 or index >= len(y):
                return math.nan
            return self._finite_number(y[index])

    def target_value(self, time_value: float | None = None) -> float:
        if time_value is None:
            time_value = float(self.network.time.value)
        return self._sample_trace_previous(float(time_value))

    def _variable_numeric_value(self) -> float:
        if not self.has_variable:
            return math.nan
        return self._finite_number(self.variable.numeric_value)

    def _hold_variable_value(self) -> float:
        if not self.has_variable:
            return math.nan

        try:
            return self._finite_number(self.variable.previous)
        except Exception:
            return self._variable_numeric_value()

    def _update_outputs(self, *, stop_on_missing: bool = False) -> float:
        target = self.target_value()
        has_target = math.isfinite(target)
        has_data = self.has_data
        has_variable = self.has_variable

        self.data_value.value = target if has_data else math.nan
        self.variable_value.value = self._variable_numeric_value()
        self.active.value = bool(has_target and has_variable and has_data)

        if has_data and not has_target:
            self.error.value = math.nan
            if stop_on_missing and not self._extend_enabled:
                raise SensorDataStop(
                    f"Sensor {self.name!r} has no finite FullPlot Trace value at "
                    f"time {float(self.network.time.value):.9g}."
                )
            return math.nan

        if has_data and has_target:
            self.error.value = self._finite_number(self.reading.value - target)
        else:
            self.error.value = math.nan

        return target

    def evaluate_states(self) -> None:
        # Pure evaluation/export should never stop a solve. The residual method
        # handles extend=False during active matching.
        self._update_outputs(stop_on_missing=False)

    @property
    def balances(self) -> list[tuple[State, Any]]:
        if not self.is_anchor:
            return []
        return [(self.variable, self.residual)]

    def residual(self) -> float:
        target = self._update_outputs(stop_on_missing=True)

        if math.isfinite(target):
            return float(self.reading.value) - target

        # extend=True: keep marching through missing data. During NaN/dropout
        # regions or outside a windowed trace, the sensor no longer has a data
        # residual. Hold the independent variable at its last accepted value so
        # it is not a free unknown in the nonlinear solve.
        hold_value = self._hold_variable_value()
        if not math.isfinite(hold_value):
            return 0.0
        return float(self.variable.numeric_value) - hold_value

    @staticmethod
    def _crossing_direction(previous_error: float, current_error: float, tolerance: float) -> str:
        if abs(current_error) <= tolerance:
            return "touch"
        if previous_error < 0.0 < current_error:
            return "upward"
        if previous_error > 0.0 > current_error:
            return "downward"
        return "crossing"

    @staticmethod
    def _crossed(previous_error: float, current_error: float, tolerance: float) -> bool:
        previous_zero = abs(previous_error) <= tolerance
        current_zero = abs(current_error) <= tolerance
        if current_zero:
            return True
        if previous_zero:
            return False
        return previous_error * current_error < 0.0

    def reset_condition_history(self, time_value: float | None = None) -> None:
        """Initialize condition crossing memory without creating events."""
        if time_value is None:
            time_value = float(self.network.time.value)
        time_value = float(time_value)
        reading = self._finite_number(self.reading.value)

        for condition in self.conditions:
            condition.previous_reading = reading
            condition.previous_line_value = self._sample_condition_trace(condition, time_value)
            condition.was_active = False

    def check_conditions(self, time_value: float | None = None) -> list[SensorEvent]:
        """Return condition-crossing events at the current accepted timestep."""
        if time_value is None:
            time_value = float(self.network.time.value)
        time_value = float(time_value)
        reading = self._finite_number(self.reading.value)
        events: list[SensorEvent] = []

        for condition in self.conditions:
            line_value = self._sample_condition_trace(condition, time_value)
            previous_reading = condition.previous_reading
            previous_line_value = condition.previous_line_value

            if not all(math.isfinite(value) for value in (reading, line_value, previous_reading, previous_line_value)):
                condition.previous_reading = reading
                condition.previous_line_value = line_value
                condition.was_active = False
                continue

            previous_error = previous_reading - previous_line_value
            current_error = reading - line_value
            tolerance = 1.0e-12 * max(
                1.0,
                abs(reading),
                abs(line_value),
                abs(previous_reading),
                abs(previous_line_value),
            )
            active = self._crossed(previous_error, current_error, tolerance)

            if active and not condition.was_active:
                crossing_direction = self._crossing_direction(previous_error, current_error, tolerance)
                message = (
                    f"{self.name} crossed {condition.name} "
                    f"({condition.role}) at t={time_value:.9g}."
                )
                events.append(
                    SensorEvent(
                        time=time_value,
                        sensor=self.name,
                        trace=condition.name,
                        role=condition.role,
                        action=condition.action,
                        reading=reading,
                        line_value=line_value,
                        previous_reading=previous_reading,
                        previous_line_value=previous_line_value,
                        crossing_direction=crossing_direction,
                        message=message,
                    )
                )

            condition.was_active = active
            condition.previous_reading = reading
            condition.previous_line_value = line_value

        return events
