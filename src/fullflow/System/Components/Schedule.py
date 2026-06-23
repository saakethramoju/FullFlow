from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any

import numpy as np

from fullflow.System import Component, State, is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Schedule(Component):
    """
    Generic sampled schedule/command source.

    `Schedule` drives `target` as a function of network time. Functional
    schedules can also read input states, but those inputs are sampled from the
    previous accepted transient step. The output is then held fixed during the
    current nonlinear solve.

    This makes Schedule safe for command logic, feedback controllers, bang-bang
    valves, pump-speed commands, and ordinary open-loop time ramps.

    The schedule can be tabular:

        Schedule(..., times=[0.0, 1.0], values=[2.0e5, 5.0e5])

    or functional:

        Schedule(..., function=source_pressure_command)

    or functional with sampled inputs:

        Schedule(..., function=valve_command, inputs=[node_pressure])

    The callable signature is:

        function(time, *previous_input_values)

    If `target` is provided, that State is overwritten.

    If `target` is not provided, Schedule creates one automatically:

        source_pressure = PressureSchedule.target
    """

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
        self._table_schedule = times is not None or values is not None
        self._function_schedule = function is not None
        self._input_list = self._normalize_inputs(inputs)

        self.setup()

        if self._table_schedule and self._function_schedule:
            raise ValueError(f"{self.name}: provide either times/values or function, not both.")

        if not self._table_schedule and not self._function_schedule:
            raise ValueError(f"{self.name}: provide either times/values or function.")

        if self._table_schedule and self._input_list:
            raise ValueError(f"{self.name}: tabular schedules cannot use inputs; use function=... instead.")

        if not is_state_like(self.target):
            raise TypeError(f"{self.name}: target must be a State-like object.")

        if self._table_schedule:
            if times is None or values is None:
                raise ValueError(f"{self.name}: tabular schedules require both times and values.")

            time_map = np.asarray(self._read(self.times), dtype=float)
            value_map = np.asarray(self._read(self.values), dtype=float)

            if time_map.ndim != 1:
                raise ValueError(f"{self.name}: times must be one-dimensional.")

            if value_map.ndim != 1:
                raise ValueError(f"{self.name}: values must be one-dimensional.")

            if len(time_map) != len(value_map):
                raise ValueError(f"{self.name}: times and values must have the same length.")

            if len(time_map) < 2:
                raise ValueError(f"{self.name}: schedule requires at least two points.")

            sort_indices = np.argsort(time_map)
            time_map = time_map[sort_indices]
            value_map = value_map[sort_indices]

            if np.any(np.diff(time_map) <= 0.0):
                raise ValueError(f"{self.name}: times must be strictly increasing.")

            self.times.value = time_map
            self.values.value = value_map

        if self._function_schedule:
            if not callable(self._read(self.function)):
                raise TypeError(f"{self.name}: function must be callable.")

        self.evaluate_in_pre_evaluation = True

        self.evaluate_states()

    def pre_evaluation(self):
        if self._read(self.evaluate_in_pre_evaluation):
            self.evaluate_states()

    def evaluate_states(self):
        t = self._network_time()

        if self._function_schedule:
            input_values = [self._previous_input_value(item) for item in self._input_list]
            value = self._read(self.function)(t, *input_values)
        else:
            value = np.interp(t, self.times.value, self.values.value)

        self.target.value = float(value)

    @property
    def transient_history_states(self) -> list[State]:
        states: list[State] = []
        seen: set[int] = set()

        for item in self._input_list:
            if not is_state_like(item):
                continue

            if not callable(getattr(item, "store_previous", None)):
                continue

            item_id = id(item)
            if item_id in seen:
                continue

            seen.add(item_id)
            states.append(item)

        return states

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

    def _read(self, variable):
        if is_state_like(variable):
            if not variable.is_assigned:
                return None

            return variable.value

        return variable
