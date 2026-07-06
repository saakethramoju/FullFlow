from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable, Any

import numpy as np
from fullplot import Trace

from fullflow.Exceptions import SolverSetupError
from fullflow.System import Component, State
from fullflow.System.State import is_assignable_state_like, is_state_like, label_state_refs

if TYPE_CHECKING:
    from fullflow.System import Network


SEQUENCE_COMMAND_ROLE = "command"
SEQUENCE_COMMAND_MISSING = {"hold", "skip", "error"}


@dataclass(slots=True)
class SequenceCommand:
    """One FullPlot command Trace applied by a Sequence."""

    target: Any
    trace: Trace
    scale: Any = 1.0
    offset: Any = 0.0
    transform: Callable[[Any], Any] | None = None
    missing: str = "hold"
    condition: Any = None
    condition_sensor: Any = None
    condition_name: str | None = None
    activation_time: float | None = 0.0
    is_active: bool = True
    last_value: Any = None
    has_last_value: bool = False
    pending_value: Any = None
    has_pending_value: bool = False

    @property
    def name(self) -> str:
        return str(getattr(self.trace, "name", "command"))

    @property
    def role(self) -> str:
        return str(getattr(self.trace, "role", SEQUENCE_COMMAND_ROLE) or SEQUENCE_COMMAND_ROLE).lower().strip()

    def clear_pending(self) -> None:
        self.pending_value = None
        self.has_pending_value = False

    def set_pending(self, value: Any) -> None:
        self.pending_value = value
        self.has_pending_value = True

    def commit(self) -> None:
        if self.has_pending_value:
            self.last_value = self.pending_value
            self.has_last_value = True
        self.clear_pending()


class Sequence(Component):
    """
    Generic transient sequence/command source.

    `Sequence` can drive one legacy target directly from a tabular or functional
    schedule, and it can also replay FullPlot command traces into any number of
    State-like targets through :meth:`command`.

    Legacy tabular use::

        Sequence(..., target=valve_area, times=[0.0, 1.0], values=[0.0, 1.0])

    Legacy functional use::

        Sequence(..., target=valve_area, function=valve_command)

    Command-trace use::

        Start = Sequence("Start Sequence", Test)
        Start.command(FuelValve.area, fuel_valve_area_command)

    Conditional command-trace use::

        Start.command(FuelValve.area, fuel_abort_command, condition=(CHPT, "High Pc"))

    Normalized command-trace use::

        Start.command(FuelValve.area, fuel_valve_open_fraction, scale=full_open_area)

    In steady-state solves, Sequence is skipped by the solver because
    ``TRANSIENT_ONLY = True``. The current target values are preserved.
    """

    TRANSIENT_ONLY = True

    def __init__(
        self,
        name: str,
        network: Network,
        target: State | None = None,
        times=None,
        values=None,
        function: Callable[..., float] | None = None,
        inputs: list[Any] | tuple[Any, ...] | Any | None = None,
    ):
        self._table_sequence = times is not None or values is not None
        self._function_sequence = function is not None
        self._has_primary_sequence = self._table_sequence or self._function_sequence
        self._input_list = self._normalize_inputs(inputs)
        self._target_was_provided = target is not None
        self._active_in_transient = False
        self._command_entries: list[SequenceCommand] = []

        self.setup()

        self.evaluate_in_pre_evaluation = True

        if self._table_sequence and self._function_sequence:
            raise ValueError(f"{self.name}: provide either times/values or function, not both.")

        if self._table_sequence and self._input_list:
            raise ValueError(f"{self.name}: tabular sequences cannot use inputs; use function=... instead.")

        if not self._has_primary_sequence:
            if self._target_was_provided:
                raise ValueError(
                    f"{self.name}: target was provided but no times/values or function was supplied. "
                    "Use Sequence.command(...) for command traces or provide a schedule."
                )
            return

        if not is_state_like(self.target):
            raise TypeError(f"{self.name}: target must be a State-like object.")

        if self._table_sequence:
            if times is None or values is None:
                raise ValueError(f"{self.name}: tabular sequences require both times and values.")

            time_map = np.asarray(self._read(self.times), dtype=float)
            value_map = np.asarray(self._read(self.values), dtype=float)

            if time_map.ndim != 1:
                raise ValueError(f"{self.name}: times must be one-dimensional.")

            if value_map.ndim != 1:
                raise ValueError(f"{self.name}: values must be one-dimensional.")

            if len(time_map) != len(value_map):
                raise ValueError(f"{self.name}: times and values must have the same length.")

            if len(time_map) < 2:
                raise ValueError(f"{self.name}: sequence requires at least two points.")

            sort_indices = np.argsort(time_map)
            time_map = time_map[sort_indices]
            value_map = value_map[sort_indices]

            if np.any(np.diff(time_map) <= 0.0):
                raise ValueError(f"{self.name}: times must be strictly increasing.")

            self.times.value = time_map
            self.values.value = value_map

        if self._function_sequence:
            if not callable(self._read(self.function)):
                raise TypeError(f"{self.name}: function must be callable.")

        if not self._target_was_provided:
            self.target.value = float(self._sequenced_value())

    @property
    def command_entries(self) -> tuple[SequenceCommand, ...]:
        """Registered command-trace entries."""
        return tuple(self._command_entries)

    @staticmethod
    def _trace_role(trace: Trace) -> str:
        return str(getattr(trace, "role", SEQUENCE_COMMAND_ROLE) or SEQUENCE_COMMAND_ROLE).lower().strip()

    @staticmethod
    def _validate_trace_object(sequence_name: str, trace: Any) -> None:
        if not isinstance(trace, Trace):
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command trace must be a fullplot.Trace object. "
                f"Got {type(trace).__name__}."
            )

    @classmethod
    def _validate_command_trace(cls, sequence_name: str, trace: Trace) -> None:
        cls._validate_trace_object(sequence_name, trace)
        role = cls._trace_role(trace)
        if role != SEQUENCE_COMMAND_ROLE:
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command trace must have role='command'. "
                f"Trace {getattr(trace, 'name', '<unnamed>')!r} has role={role!r}."
            )

    @staticmethod
    def _trace_arrays_for(trace: Trace, sequence_name: str) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(getattr(trace, "x", []), dtype=float)
        y_array = np.asarray(getattr(trace, "y", []), dtype=float)

        if x_array.ndim != 1 or y_array.ndim != 1:
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command Trace {getattr(trace, 'name', '<unnamed>')!r} "
                "x and y arrays must be one-dimensional."
            )

        if x_array.shape != y_array.shape:
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command Trace {getattr(trace, 'name', '<unnamed>')!r} "
                f"x and y arrays must have the same length. Got {len(x_array)} and {len(y_array)}."
            )

        if len(x_array) == 0:
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command Trace {getattr(trace, 'name', '<unnamed>')!r} is empty."
            )

        if not np.all(np.isfinite(x_array)):
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command Trace {getattr(trace, 'name', '<unnamed>')!r} "
                "time values must be finite."
            )

        if len(x_array) > 1 and np.any(np.diff(x_array) <= 0.0):
            raise SolverSetupError(
                f"Sequence {sequence_name!r} command Trace {getattr(trace, 'name', '<unnamed>')!r} "
                "time values must be strictly increasing."
            )

        return x_array, y_array

    def command(
        self,
        target: Any = None,
        trace: Trace | None = None,
        *,
        condition: Any = None,
        scale: Any = 1.0,
        offset: Any = 0.0,
        transform: Callable[[Any], Any] | None = None,
        missing: str = "hold",
    ) -> SequenceCommand:
        """Replay a FullPlot command Trace into an assignable State-like target.

        The common case is direct replay::

            sequence.command(FuelValve.area, fuel_valve_area_command)

        Conditional commands switch to another command trace when a Sensor
        condition is crossed::

            sequence.command(FuelValve.area, fuel_abort_command, condition=(CHPT, "High Pc"))

        ``condition`` may be ``None`` for an immediate command, a number for an
        absolute start time, or ``(sensor, condition_name)`` for a Sensor event.
        Command Trace time is relative to its activation time.

        For normalized commands, use ``scale`` and optional ``offset``::

            target.value = offset + scale * trace_value

        ``transform`` is an advanced escape hatch for custom conversions. It is
        applied after ``scale`` and ``offset``.

        ``missing`` controls NaN or out-of-range command samples:

        ``"hold"``
            Reuse the last accepted finite command value. If no valid command
            has been accepted yet, leave the target unchanged.

        ``"skip"``
            Leave the target unchanged.

        ``"error"``
            Raise a clear error when a command value is missing.
        """
        if target is None:
            raise SolverSetupError(f"Sequence {self.name!r} command target is required.")
        if trace is None:
            raise SolverSetupError(f"Sequence {self.name!r} command trace is required.")
        self._validate_command_trace(self.name, trace)

        if not is_assignable_state_like(target):
            raise SolverSetupError(
                f"Sequence {self.name!r} command target must be an assignable State-like object. "
                f"Got {type(target).__name__}."
            )

        if transform is not None and not callable(transform):
            raise SolverSetupError(f"Sequence {self.name!r} command transform must be callable or None.")

        missing = str(missing).lower().strip()
        if missing not in SEQUENCE_COMMAND_MISSING:
            valid = ", ".join(repr(item) for item in sorted(SEQUENCE_COMMAND_MISSING))
            raise SolverSetupError(
                f"Sequence {self.name!r} command missing must be one of {valid}. Got {missing!r}."
            )

        self._trace_arrays_for(trace, self.name)
        condition_sensor, condition_name, activation_time, is_active = self._normalize_command_condition(condition)
        label_state_refs(target, f"{self.name}:command:{getattr(trace, 'name', 'command')}")

        entry = SequenceCommand(
            target=target,
            trace=trace,
            scale=scale,
            offset=offset,
            transform=transform,
            missing=missing,
            condition=condition,
            condition_sensor=condition_sensor,
            condition_name=condition_name,
            activation_time=activation_time,
            is_active=is_active,
        )
        self._command_entries.append(entry)
        self.network.mark_structure_changed()
        return entry

    def set_transient_context(self, *, dt: float) -> None:
        super().set_transient_context(dt=dt)
        self._active_in_transient = True

    def pre_evaluation(self):
        if self._active_in_transient and self._read(self.evaluate_in_pre_evaluation):
            self.apply_commands()

    def evaluate_states(self):
        if not self._active_in_transient:
            return

        self.apply_commands()

    def apply_commands(self, time_value: float | None = None) -> None:
        """Apply this Sequence's commands at the current transient time."""
        if not self._active_in_transient:
            return

        if time_value is None:
            time_value = self._network_time()
        time_value = float(time_value)

        if self._has_primary_sequence:
            self.target.value = float(self._sequenced_value(time_value=time_value))

        for entry in self._command_entries:
            self._apply_command(entry, time_value)

    def commit_commands(self) -> None:
        """Commit valid command samples after an accepted timestep."""
        for entry in self._command_entries:
            entry.commit()

    def clear_command_pending(self) -> None:
        """Clear unaccepted command samples."""
        for entry in self._command_entries:
            entry.clear_pending()

    def command_breakpoints(self) -> tuple[float, ...]:
        """Return finite command Trace time points for timestep snapping."""
        breakpoints: set[float] = set()

        for entry in self._command_entries:
            try:
                x_array, _ = self._trace_arrays_for(entry.trace, self.name)
            except Exception:
                continue

            offset = 0.0
            if entry.condition_sensor is None and entry.activation_time is not None:
                offset = float(entry.activation_time)
            elif entry.is_active and entry.activation_time is not None:
                offset = float(entry.activation_time)
            else:
                continue

            for time_value in x_array:
                value = offset + float(time_value)
                if math.isfinite(value):
                    breakpoints.add(value)

        return tuple(sorted(breakpoints))

    def command_trace_records(self) -> list[dict[str, Any]]:
        """Return command Trace definitions for HDF5 export."""
        rows: list[dict[str, Any]] = []

        for entry in self._command_entries:
            try:
                x_array, y_array = self._trace_arrays_for(entry.trace, self.name)
            except Exception:
                x_array = np.asarray(getattr(entry.trace, "x", []), dtype=float)
                y_array = np.asarray(getattr(entry.trace, "y", []), dtype=float)

            rows.append(
                {
                    "sequence": self.name,
                    "trace": entry.name,
                    "role": entry.role,
                    "target": self._target_label(entry.target),
                    "missing": entry.missing,
                    "condition": self._condition_label(entry),
                    "condition_sensor": getattr(entry.condition_sensor, "name", "") if entry.condition_sensor is not None else "",
                    "condition_name": entry.condition_name or "",
                    "activation_time": math.nan if entry.activation_time is None else float(entry.activation_time),
                    "scale": self._read(entry.scale),
                    "offset": self._read(entry.offset),
                    "has_transform": entry.transform is not None,
                    "x": x_array,
                    "y": y_array,
                }
            )

        return rows

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (bool, np.bool_)):
            return False
        try:
            return not math.isfinite(float(value))
        except Exception:
            return False

    def activate_condition_commands(self, events: list[Any]) -> None:
        """Activate commands whose Sensor condition occurred on an accepted step."""
        if not events:
            return

        for entry in self._command_entries:
            if entry.condition_sensor is None or entry.condition_name is None or entry.is_active:
                continue

            sensor_name = str(getattr(entry.condition_sensor, "name", ""))
            for event in events:
                if str(getattr(event, "sensor", "")) != sensor_name:
                    continue
                if str(getattr(event, "trace", "")) != str(entry.condition_name):
                    continue
                try:
                    activation_time = float(getattr(event, "time"))
                except Exception:
                    activation_time = self._network_time()
                entry.activation_time = activation_time
                entry.is_active = True
                entry.clear_pending()
                entry.has_last_value = False
                entry.last_value = None
                break

    def _normalize_command_condition(self, condition: Any) -> tuple[Any, str | None, float | None, bool]:
        if condition is None:
            return None, None, 0.0, True

        if isinstance(condition, (int, float, np.integer, np.floating)):
            start_time = float(condition)
            if not math.isfinite(start_time):
                raise SolverSetupError(f"Sequence {self.name!r} command condition time must be finite.")
            return None, None, start_time, False

        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            sensor, condition_name = condition
            condition_name = str(condition_name)
            has_condition = getattr(type(sensor), "has_condition", None)
            if not callable(has_condition):
                raise SolverSetupError(
                    f"Sequence {self.name!r} command condition must be (Sensor, condition_name). "
                    f"Got {type(sensor).__name__}."
                )
            if not has_condition(sensor, condition_name):
                raise SolverSetupError(
                    f"Sequence {self.name!r} command condition {condition_name!r} was not found on "
                    f"Sensor {getattr(sensor, 'name', '<unnamed>')!r}."
                )
            return sensor, condition_name, None, False

        raise SolverSetupError(
            f"Sequence {self.name!r} command condition must be None, a start time, "
            "or (Sensor, condition_name)."
        )

    def _command_is_active(self, entry: SequenceCommand, time_value: float) -> bool:
        if entry.condition_sensor is not None:
            return bool(entry.is_active and entry.activation_time is not None)

        if entry.activation_time is None:
            entry.activation_time = 0.0

        if not entry.is_active and time_value >= float(entry.activation_time):
            entry.is_active = True

        return bool(entry.is_active)

    @staticmethod
    def _condition_label(entry: SequenceCommand) -> str:
        if entry.condition_sensor is not None and entry.condition_name is not None:
            return f"{getattr(entry.condition_sensor, 'name', 'sensor')}:{entry.condition_name}"
        if entry.activation_time is not None and float(entry.activation_time) != 0.0:
            return f"time:{float(entry.activation_time):.9g}"
        return "immediate"

    def _apply_command(self, entry: SequenceCommand, time_value: float) -> None:
        if not self._command_is_active(entry, time_value):
            entry.clear_pending()
            return

        local_time = time_value - float(entry.activation_time or 0.0)
        raw_value = self._sample_command_trace(entry.trace, local_time)

        if self._is_missing(raw_value):
            entry.clear_pending()
            self._handle_missing_command(entry, time_value)
            return

        value = self._converted_command_value(entry, raw_value)
        if self._is_missing(value):
            entry.clear_pending()
            self._handle_missing_command(entry, time_value)
            return

        entry.target.value = value
        entry.set_pending(value)

    def _handle_missing_command(self, entry: SequenceCommand, time_value: float) -> None:
        if entry.missing == "hold":
            if entry.has_last_value:
                entry.target.value = entry.last_value
            return

        if entry.missing == "skip":
            return

        raise SolverSetupError(
            f"Sequence {self.name!r} command Trace {entry.name!r} has no finite command value "
            f"at time {time_value:.9g}."
        )

    def _converted_command_value(self, entry: SequenceCommand, raw_value: Any) -> Any:
        scale = self._read(entry.scale)
        offset = self._read(entry.offset)

        if self._is_missing(scale) or self._is_missing(offset):
            return math.nan

        try:
            value = offset + scale * raw_value
        except Exception:
            return math.nan

        if entry.transform is not None:
            try:
                value = entry.transform(value)
            except Exception:
                return math.nan

        return value

    def _sample_command_trace(self, trace: Trace, time_value: float) -> float:
        if not math.isfinite(time_value):
            return math.nan

        x_array, y_array = self._trace_arrays_for(trace, self.name)

        if time_value < x_array[0] or time_value > x_array[-1]:
            return math.nan

        if len(x_array) == 1:
            return self._finite_number(y_array[0]) if time_value == x_array[0] else math.nan

        return self._finite_number(np.interp(time_value, x_array, y_array))

    def _sequenced_value(self, time_value: float | None = None):
        t = self._network_time() if time_value is None else float(time_value)

        if self._function_sequence:
            input_values = [self._previous_input_value(item) for item in self._input_list]
            return self._read(self.function)(t, *input_values)

        return np.interp(t, self.times.value, self.values.value)

    @staticmethod
    def _normalize_inputs(inputs):
        if inputs is None:
            return []

        if is_state_like(inputs):
            return [inputs]

        if isinstance(inputs, (list, tuple)):
            return list(inputs)

        return [inputs]

    def _network_time(self):
        if not hasattr(self.network, "time"):
            return 0.0

        if is_state_like(self.network.time):
            if not self.network.time.is_assigned:
                return 0.0

            return float(self.network.time.value)

        return float(self.network.time)

    def _previous_input_value(self, variable):
        if is_state_like(variable):
            try:
                return variable.previous
            except Exception:
                if variable.is_assigned:
                    return variable.value
                return None

        return variable

    @staticmethod
    def _finite_number(value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return math.nan
        return number if math.isfinite(number) else math.nan

    @staticmethod
    def _target_label(target: Any) -> str:
        labels = getattr(target, "labels", None)
        if labels:
            try:
                return str(tuple(labels)[0])
            except Exception:
                pass
        label = getattr(target, "label", None)
        if label is not None:
            return str(label)
        return str(target)

    def _read(self, variable):
        if is_state_like(variable):
            if not variable.is_assigned:
                return None

            return variable.value

        return variable
