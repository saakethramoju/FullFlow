from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import inspect
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


@dataclass(frozen=True, slots=True)
class SequenceCondition:
    """Reference to one named ``Sensor`` condition used by a sequence command or abort."""

    sensor: Any
    name: str

    @property
    def sensor_name(self) -> str:
        return str(getattr(self.sensor, "name", ""))

    @property
    def key(self) -> tuple[str, str]:
        return (self.sensor_name, str(self.name))

    @property
    def label(self) -> str:
        return f"{self.sensor_name}:{self.name}"


@dataclass(slots=True)
class SequenceCommand:
    """One pending or committed command managed by a ``Sequence``."""

    target: Any
    trace: Trace
    scale: Any = 1.0
    offset: Any = 0.0
    transform: Callable[[Any], Any] | None = None
    missing: str = "hold"
    condition: Any = None
    condition_sensor: Any = None
    condition_name: str | None = None
    condition_terms: tuple[SequenceCondition, ...] = ()
    satisfied_condition_keys: set[tuple[str, str]] = field(default_factory=set)
    satisfied_condition_times: dict[tuple[str, str], float] = field(default_factory=dict)
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


@dataclass(slots=True)
class SequenceAbort:
    """Clean transient stop condition managed by a ``Sequence``."""

    condition: Any = None
    condition_sensor: Any = None
    condition_name: str | None = None
    condition_terms: tuple[SequenceCondition, ...] = ()
    satisfied_condition_keys: set[tuple[str, str]] = field(default_factory=set)
    satisfied_condition_times: dict[tuple[str, str], float] = field(default_factory=dict)
    trigger_time: float | None = None
    abort_time: float | None = None
    delay: float = 0.0
    message: str | None = None
    is_triggered: bool = False
    has_aborted: bool = False


class Sequence(Component):
    """Transient schedules and user-defined feedback or switching control."""

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
        event: bool = False,
        sample_rate: float | None = None,
        memory: Any = None,
    ):
        """Initialize the object and register any FullFlow state wiring."""
        self._event_enabled = bool(event)
        self._event_command = None
        self._step_held = False
        self._sample_enabled = sample_rate is not None
        if self._event_enabled and function is None:
            raise ValueError(f"{name}: event=True requires a function-based Sequence.")
        if self._sample_enabled:
            rate = float(sample_rate)
            if not math.isfinite(rate) or rate <= 0.0 or not math.isfinite(1.0 / rate):
                raise ValueError(f"{name}: sample_rate must be a finite positive frequency in Hz.")
            if rate > 1.0e10:
                raise ValueError(f"{name}: sample_rate is too high for reliable floating-point scheduling.")
            if function is None or event:
                raise ValueError(f"{name}: sampled Sequences require function=... and cannot use event=True.")
            if target is None:
                raise ValueError(f"{name}: sampled Sequences require an initialized target State.")
            self._sample_period = 1.0 / rate
        else:
            if memory is not None:
                raise ValueError(f"{name}: memory requires sample_rate.")
            self._sample_period = None
        self._sample_memory = deepcopy(memory)
        self._sample_origin = None
        self._sample_index = 1
        self._sample_command = None
        self._sample_pass_dt = False
        self._sample_pass_memory = False
        self._table_sequence = times is not None or values is not None
        self._function_sequence = function is not None
        self._has_primary_sequence = self._table_sequence or self._function_sequence
        self._input_list = self._normalize_inputs(inputs)
        self._target_was_provided = target is not None
        self._active_in_transient = False
        self._command_entries: list[SequenceCommand] = []
        self._abort_entries: list[SequenceAbort] = []

        self.setup()
        # Scheduling and memory are internal, not solver State inputs.
        del self.event
        del self.sample_rate
        del self.memory

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
            if not self._event_enabled and not self._sample_enabled and any(item is self.target for item in self._input_list):
                self._step_held = True
            if self._sample_enabled:
                try:
                    signature = inspect.signature(self._read(self.function))
                    params = signature.parameters
                    flexible = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                    self._sample_pass_dt = flexible or 'dt' in params
                    self._sample_pass_memory = flexible or 'memory' in params or memory is not None
                except (TypeError, ValueError):
                    self._sample_pass_dt = memory is not None
                    self._sample_pass_memory = memory is not None
                if not self.target.is_assigned:
                    raise ValueError(f"{self.name}: sampled target State must have an initial command value.")
                self._sample_command = float(self.target.value)
                if not math.isfinite(self._sample_command):
                    raise ValueError(f"{self.name}: sampled target State must have a finite initial command.")

        if not self._target_was_provided:
            self.target.value = float(self._sequenced_value())

    @property
    def command_entries(self) -> tuple[SequenceCommand, ...]:
        """Registered command-trace entries."""
        return tuple(self._command_entries)

    @property
    def abort_entries(self) -> tuple[SequenceAbort, ...]:
        """Registered clean transient abort rules."""
        return tuple(self._abort_entries)

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
        """Replay a FullPlot command Trace into an assignable State-like target."""
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
        condition_sensor, condition_name, condition_terms, activation_time, is_active = self._normalize_command_condition(condition)
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
            condition_terms=condition_terms,
            activation_time=activation_time,
            is_active=is_active,
        )
        self._command_entries.append(entry)
        self.network.mark_structure_changed()
        return entry

    def abort(
        self,
        condition: Any = None,
        *,
        delay: float = 0.0,
        message: str | None = None,
    ) -> SequenceAbort:
        """Stop the transient cleanly when a Sequence condition is satisfied."""
        try:
            delay = float(delay)
        except Exception as error:
            raise SolverSetupError(f"Sequence {self.name!r} abort delay must be numeric.") from error

        if not math.isfinite(delay) or delay < 0.0:
            raise SolverSetupError(f"Sequence {self.name!r} abort delay must be finite and non-negative.")

        condition_sensor, condition_name, condition_terms, activation_time, is_active = self._normalize_command_condition(condition)

        abort_time = None
        trigger_time = None
        is_triggered = False

        if condition_sensor is None:
            trigger_time = 0.0 if activation_time is None else float(activation_time)
            abort_time = trigger_time + delay
            is_triggered = True

        entry = SequenceAbort(
            condition=condition,
            condition_sensor=condition_sensor,
            condition_name=condition_name,
            condition_terms=condition_terms,
            trigger_time=trigger_time,
            abort_time=abort_time,
            delay=delay,
            message=message,
            is_triggered=is_triggered,
        )
        self._abort_entries.append(entry)
        self.network.mark_structure_changed()
        return entry

    def set_transient_context(self, *, dt: float) -> None:
        """Receive timestep-level information from the transient solver."""
        super().set_transient_context(dt=dt)
        self._active_in_transient = True

    def pre_evaluation(self):
        """Run pre-residual bookkeeping before component evaluation."""
        if self._active_in_transient and self._read(self.evaluate_in_pre_evaluation):
            self.apply_commands()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
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
            if self._sample_enabled:
                self.target.value = self._sample_command
            elif self._event_enabled:
                if self._event_command is None:
                    self._event_command = float(self._sequenced_value(time_value=time_value))
                self.target.value = self._event_command
            elif self._step_held:
                self.target.value = self._previous_input_value(self.target)
            else:
                self.target.value = float(self._sequenced_value(time_value=time_value))

        for entry in self._command_entries:
            self._apply_command(entry, time_value)

    def initialize_sampling(self, time_value: float) -> None:
        """Start the clock at the accepted initial point; hold the initial target."""
        if not self._sample_enabled:
            return
        self._sample_origin = float(time_value)
        self._sample_index = 1
        self._sample_command = float(self.target.value)

    def next_sample_time(self) -> float | None:
        """Compute the next exact clock time without accumulating time drift."""
        if self._sample_origin is None:
            return None
        return self._sample_origin + self._sample_index * self._sample_period

    def sample_due(self, time_value: float) -> bool:
        next_time = self.next_sample_time()
        if next_time is None:
            return False
        tolerance = 1e-12 * max(1.0, abs(next_time), abs(time_value))
        return abs(time_value - next_time) <= tolerance

    def prepare_sample(self, time_value: float):
        """Evaluate once from an accepted plant state, without committing memory."""
        if not self.sample_due(time_value):
            return None
        measured = [self._read(item) for item in self._input_list]
        kwargs = {}
        if self._sample_pass_dt:
            kwargs['dt'] = self._sample_period
        if self._sample_pass_memory:
            kwargs['memory'] = deepcopy(self._sample_memory)
        result = self._read(self.function)(float(time_value), *measured, **kwargs)
        if self._sample_pass_memory:
            if not isinstance(result, tuple) or len(result) != 2:
                raise SolverSetupError(
                    f"{self.name}: stateful sampled function must return (command, new_memory)."
                )
            command, new_memory = result
        else:
            command, new_memory = result, None
        command = float(command)
        if not math.isfinite(command):
            raise SolverSetupError(f"{self.name}: sampled function returned a nonfinite command.")
        return command, deepcopy(new_memory)

    def accept_sample(self, value, memory) -> None:
        """Commit exactly one accepted sample, with no extra nonlinear solve."""
        self._sample_command = value
        self.target.value = value
        self._sample_memory = memory
        self._sample_index += 1

    def initialize_event(self) -> None:
        """Latch an initial discrete command from the current accepted State."""
        if not self._event_enabled:
            return
        self._event_command = None
        self._event_command = self.event_request(float(self._network_time()))
        self.target.value = self._event_command

    def _event_probe_inputs(self):
        """Return current guard inputs without changing the latched command."""
        return tuple(
            self._event_command if item is self.target and self._event_command is not None
            else self._read(item)
            for item in self._input_list
        )

    def _event_request_from_inputs(self, time_value, values):
        """Probe a discrete function using supplied values, without changing States."""
        value = float(self._read(self.function)(float(time_value), *values))
        if not math.isfinite(value):
            raise SolverSetupError(f"{self.name}: event function returned a nonfinite command.")
        return value

    def event_request(self, time_value: float) -> float:
        """Evaluate a pure discrete function at a converged candidate solution."""
        if not self._event_enabled:
            raise ValueError("Only event-enabled functions have event requests.")
        return self._event_request_from_inputs(time_value, self._event_probe_inputs())

    def accept_event(self, command: float) -> None:
        """Commit the discrete mode exactly once at a located event boundary."""
        if not self._event_enabled:
            raise ValueError("Only event-enabled functions can accept an event.")
        self._event_command = float(command)
        self.target.value = self._event_command

    def commit_commands(self) -> None:
        """Commit valid command samples after an accepted timestep."""
        if self._step_held:
            self.target.value = float(self._sequenced_value())
            self.target.store_previous()
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
                    "condition_sensor": self._condition_sensor_label(entry),
                    "condition_name": self._condition_name_label(entry),
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
        """Activate commands whose Sensor conditions occurred on accepted steps."""
        if not events:
            return

        event_times = self._event_time_map(events)

        for entry in self._command_entries:
            if not entry.condition_terms or entry.is_active:
                continue

            self._record_satisfied_conditions(
                terms=entry.condition_terms,
                satisfied_keys=entry.satisfied_condition_keys,
                satisfied_times=entry.satisfied_condition_times,
                event_times=event_times,
            )

            if not self._all_conditions_satisfied(entry.condition_terms, entry.satisfied_condition_keys):
                continue

            activation_time = max(entry.satisfied_condition_times.values(), default=self._network_time())
            entry.activation_time = activation_time
            entry.is_active = True
            entry.clear_pending()
            entry.has_last_value = False
            entry.last_value = None

    def activate_condition_aborts(self, events: list[Any]) -> None:
        """Arm clean abort rules whose Sensor conditions occurred on accepted steps."""
        if not events:
            return

        event_times = self._event_time_map(events)

        for entry in self._abort_entries:
            if not entry.condition_terms or entry.is_triggered:
                continue

            self._record_satisfied_conditions(
                terms=entry.condition_terms,
                satisfied_keys=entry.satisfied_condition_keys,
                satisfied_times=entry.satisfied_condition_times,
                event_times=event_times,
            )

            if not self._all_conditions_satisfied(entry.condition_terms, entry.satisfied_condition_keys):
                continue

            trigger_time = max(entry.satisfied_condition_times.values(), default=self._network_time())
            entry.trigger_time = trigger_time
            entry.abort_time = trigger_time + float(entry.delay)
            entry.is_triggered = True

    def next_abort_time(self, current_time: float | None = None) -> float | None:
        """Return the next clean abort time requested by this Sequence."""
        if current_time is None:
            current_time = self._network_time()
        current_time = float(current_time)
        tolerance = 1.0e-12 * max(1.0, abs(current_time))

        candidates: list[float] = []
        for entry in self._abort_entries:
            if entry.has_aborted or not entry.is_triggered or entry.abort_time is None:
                continue
            abort_time = float(entry.abort_time)
            if math.isfinite(abort_time) and abort_time >= current_time - tolerance:
                candidates.append(abort_time)

        if not candidates:
            return None
        return min(candidates)

    def check_abort(self, time_value: float | None = None) -> dict[str, Any] | None:
        """Return an abort record if any clean abort rule is due."""
        if time_value is None:
            time_value = self._network_time()
        time_value = float(time_value)
        tolerance = 1.0e-12 * max(1.0, abs(time_value))

        due_entries = []
        for entry in self._abort_entries:
            if entry.has_aborted or not entry.is_triggered or entry.abort_time is None:
                continue
            abort_time = float(entry.abort_time)
            if math.isfinite(abort_time) and abort_time <= time_value + tolerance:
                due_entries.append(entry)

        if not due_entries:
            return None

        entry = min(due_entries, key=lambda item: float(item.abort_time or time_value))
        entry.has_aborted = True
        return {
            "status": "aborted",
            "sequence": self.name,
            "time": float(entry.abort_time if entry.abort_time is not None else time_value),
            "trigger_time": math.nan if entry.trigger_time is None else float(entry.trigger_time),
            "delay": float(entry.delay),
            "condition": self._abort_condition_label(entry),
            "condition_sensor": self._abort_condition_sensor_label(entry),
            "condition_name": self._abort_condition_name_label(entry),
            "message": entry.message or f"Sequence {self.name!r} requested abort.",
        }

    def reset_command_history(self) -> None:
        """Reset one-run command activation memory without changing configured commands."""
        for entry in self._command_entries:
            entry.clear_pending()
            entry.has_last_value = False
            entry.last_value = None
            entry.satisfied_condition_keys.clear()
            entry.satisfied_condition_times.clear()

            if entry.condition_terms:
                entry.is_active = False
                entry.activation_time = None
            elif entry.activation_time is None:
                entry.activation_time = 0.0
                entry.is_active = True
            elif float(entry.activation_time) == 0.0:
                entry.is_active = True
            else:
                entry.is_active = False

    def reset_abort_history(self) -> None:
        """Reset one-run abort state without changing configured abort rules."""
        for entry in self._abort_entries:
            entry.has_aborted = False
            if entry.condition_terms:
                entry.is_triggered = False
                entry.trigger_time = None
                entry.abort_time = None
                entry.satisfied_condition_keys.clear()
                entry.satisfied_condition_times.clear()
            else:
                start_time = 0.0 if entry.trigger_time is None else float(entry.trigger_time)
                entry.trigger_time = start_time
                entry.abort_time = start_time + float(entry.delay)
                entry.is_triggered = True

    @staticmethod
    def _abort_condition_label(entry: SequenceAbort) -> str:
        if entry.condition_terms:
            if len(entry.condition_terms) == 1:
                return entry.condition_terms[0].label
            labels = ", ".join(term.label for term in entry.condition_terms)
            return f"all({labels})"
        if entry.trigger_time is not None and float(entry.trigger_time) != 0.0:
            return f"time:{float(entry.trigger_time):.9g}"
        return "immediate"

    @staticmethod
    def _abort_condition_sensor_label(entry: SequenceAbort) -> str:
        return ",".join(term.sensor_name for term in entry.condition_terms)

    @staticmethod
    def _abort_condition_name_label(entry: SequenceAbort) -> str:
        return ",".join(str(term.name) for term in entry.condition_terms)

    @staticmethod
    def _is_single_sensor_condition(condition: Any) -> bool:
        if not isinstance(condition, (tuple, list)) or len(condition) != 2:
            return False
        sensor, _ = condition
        return callable(getattr(type(sensor), "has_condition", None))

    def _make_condition_term(self, item: Any) -> SequenceCondition:
        if not self._is_single_sensor_condition(item):
            raise SolverSetupError(
                f"Sequence {self.name!r} condition entries must be (Sensor, condition_name). "
                f"Got {type(item).__name__}."
            )

        sensor, condition_name = item
        condition_name = str(condition_name)
        has_condition = getattr(type(sensor), "has_condition", None)
        if not has_condition(sensor, condition_name):
            raise SolverSetupError(
                f"Sequence {self.name!r} command condition {condition_name!r} was not found on "
                f"Sensor {getattr(sensor, 'name', '<unnamed>')!r}."
            )
        return SequenceCondition(sensor=sensor, name=condition_name)

    def _normalize_condition_terms(self, condition: Any) -> tuple[SequenceCondition, ...]:
        if self._is_single_sensor_condition(condition):
            return (self._make_condition_term(condition),)

        if isinstance(condition, (tuple, list)) and condition:
            terms = tuple(self._make_condition_term(item) for item in condition)
            seen: set[tuple[str, str]] = set()
            for term in terms:
                if term.key in seen:
                    raise SolverSetupError(
                        f"Sequence {self.name!r} condition {term.label!r} was listed more than once."
                    )
                seen.add(term.key)
            return terms

        raise SolverSetupError(
            f"Sequence {self.name!r} command condition must be None, a start time, "
            "(Sensor, condition_name), or a list/tuple of (Sensor, condition_name) pairs."
        )

    def _normalize_command_condition(self, condition: Any) -> tuple[Any, str | None, tuple[SequenceCondition, ...], float | None, bool]:
        if condition is None:
            return None, None, (), 0.0, True

        if isinstance(condition, (int, float, np.integer, np.floating)):
            start_time = float(condition)
            if not math.isfinite(start_time):
                raise SolverSetupError(f"Sequence {self.name!r} command condition time must be finite.")
            return None, None, (), start_time, False

        terms = self._normalize_condition_terms(condition)
        if len(terms) == 1:
            return terms[0].sensor, terms[0].name, terms, None, False
        return None, None, terms, None, False

    @staticmethod
    def _event_time_map(events: list[Any]) -> dict[tuple[str, str], float]:
        event_times: dict[tuple[str, str], float] = {}
        for event in events:
            key = (str(getattr(event, "sensor", "")), str(getattr(event, "trace", "")))
            try:
                event_time = float(getattr(event, "time"))
            except Exception:
                event_time = math.nan
            if not math.isfinite(event_time):
                continue
            previous = event_times.get(key)
            if previous is None or event_time > previous:
                event_times[key] = event_time
        return event_times

    @staticmethod
    def _record_satisfied_conditions(
        *,
        terms: tuple[SequenceCondition, ...],
        satisfied_keys: set[tuple[str, str]],
        satisfied_times: dict[tuple[str, str], float],
        event_times: dict[tuple[str, str], float],
    ) -> None:
        for term in terms:
            event_time = event_times.get(term.key)
            if event_time is None:
                continue
            satisfied_keys.add(term.key)
            satisfied_times[term.key] = event_time

    @staticmethod
    def _all_conditions_satisfied(
        terms: tuple[SequenceCondition, ...],
        satisfied_keys: set[tuple[str, str]],
    ) -> bool:
        return all(term.key in satisfied_keys for term in terms)

    def _command_is_active(self, entry: SequenceCommand, time_value: float) -> bool:
        if entry.condition_terms:
            return bool(entry.is_active and entry.activation_time is not None)

        if entry.activation_time is None:
            entry.activation_time = 0.0

        if not entry.is_active and time_value >= float(entry.activation_time):
            entry.is_active = True

        return bool(entry.is_active)

    @staticmethod
    def _condition_label(entry: SequenceCommand) -> str:
        if entry.condition_terms:
            if len(entry.condition_terms) == 1:
                return entry.condition_terms[0].label
            labels = ", ".join(term.label for term in entry.condition_terms)
            return f"all({labels})"
        if entry.activation_time is not None and float(entry.activation_time) != 0.0:
            return f"time:{float(entry.activation_time):.9g}"
        return "immediate"

    @staticmethod
    def _condition_sensor_label(entry: SequenceCommand) -> str:
        return ",".join(term.sensor_name for term in entry.condition_terms)

    @staticmethod
    def _condition_name_label(entry: SequenceCommand) -> str:
        return ",".join(str(term.name) for term in entry.condition_terms)

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
            input_values = [self._feedback_input_value(item) for item in self._input_list]
            return self._read(self.function)(t, *input_values)

        return np.interp(t, self.times.value, self.values.value)

    @staticmethod
    def _normalize_input(value):
        if isinstance(value, State):
            return value

        as_state = getattr(type(value), "as_state", None)
        if callable(as_state):
            return as_state(value)

        return value

    @classmethod
    def _normalize_inputs(cls, inputs):
        if inputs is None:
            return []

        if is_state_like(inputs):
            inputs = [inputs]
        elif isinstance(inputs, (list, tuple)):
            inputs = list(inputs)
        else:
            inputs = [inputs]

        return [cls._normalize_input(value) for value in inputs]

    def _network_time(self):
        if not hasattr(self.network, "time"):
            return 0.0

        if is_state_like(self.network.time):
            if not self.network.time.is_assigned:
                return 0.0

            return float(self.network.time.value)

        return float(self.network.time)

    def _feedback_input_value(self, variable):
        """Read the trial plant State, but retain accepted self-feedback."""
        if variable is self.target:
            if self._event_enabled and self._event_command is not None:
                return self._event_command
            return self._previous_input_value(variable)
        current = self._read(variable)
        if current is None and is_state_like(variable):
            return self._previous_input_value(variable)
        return current

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
