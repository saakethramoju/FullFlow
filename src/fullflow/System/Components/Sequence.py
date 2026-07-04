from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any

import numpy as np

from fullflow.System import Component, State, is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Sequence(Component):
    """
    Generic sampled sequence/command source.

    `Sequence` drives `target` as a function of network time during transient
    solves. Functional sequences can also read input states, but those inputs
    are sampled from the previous accepted transient step.

    In steady-state solves, Sequence is skipped by the solver because
    ``TRANSIENT_ONLY = True``. The target State keeps its
    current value. This lets a steady-state solve use a fixed initial command
    such as a closed valve, while the transient solve later updates that command
    from the sequence.

    The sequence can be tabular:

        Sequence(..., times=[0.0, 1.0], values=[2.0e5, 5.0e5])

    or functional:

        Sequence(..., function=source_pressure_command)

    or functional with sampled inputs:

        Sequence(..., function=valve_command, inputs=[node_pressure])

    The callable signature is:

        function(time, *previous_input_values)

    If `target` is provided, that State is driven during transient solves.

    If `target` is not provided, Sequence creates one automatically and seeds it
    from the initial sequence value:

        source_pressure = PressureSequence.target
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
        self._input_list = self._normalize_inputs(inputs)
        self._target_was_provided = target is not None
        self._active_in_transient = False

        self.setup()

        if self._table_sequence and self._function_sequence:
            raise ValueError(f"{self.name}: provide either times/values or function, not both.")

        if not self._table_sequence and not self._function_sequence:
            raise ValueError(f"{self.name}: provide either times/values or function.")

        if self._table_sequence and self._input_list:
            raise ValueError(f"{self.name}: tabular sequences cannot use inputs; use function=... instead.")

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

        self.evaluate_in_pre_evaluation = True

        if not self._target_was_provided:
            self.target.value = float(self._sequenced_value())

    def set_transient_context(self, *, dt: float) -> None:
        super().set_transient_context(dt=dt)
        self._active_in_transient = True

    def pre_evaluation(self):
        if self._active_in_transient and self._read(self.evaluate_in_pre_evaluation):
            self.evaluate_states()

    def evaluate_states(self):
        if not self._active_in_transient:
            return

        self.target.value = float(self._sequenced_value())

    def _sequenced_value(self):
        t = self._network_time()

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

    def _read(self, variable):
        if is_state_like(variable):
            if not variable.is_assigned:
                return None

            return variable.value

        return variable