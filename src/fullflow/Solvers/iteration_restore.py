"""Restore only solver-owned States written by a component evaluation.

Only exact FullFlow States use write tracking. Custom State-like proxies and
State subclasses retain the original full restoration semantics.
"""

import math

from fullflow.System.State import State


class IterationRestorer:
    __slots__ = ("active", "_states", "_values", "_touched", "_seen", "_fallback", "_previous")

    def __init__(self, snapshot):
        self.active = False
        self._states = []
        self._values = {}
        self._touched = []
        self._seen = set()
        self._fallback = []
        self._previous = []
        track_writes = len(snapshot) >= 12
        for state, value in snapshot:
            if track_writes and type(state) is State and state._expr is None:
                self._states.append(state)
                self._values[id(state)] = value
            else:
                self._fallback.append((state, value))

    def __enter__(self):
        for state in self._states:
            self._previous.append((state, state._iteration_observer))
            state._iteration_observer = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        for state, previous in reversed(self._previous):
            state._iteration_observer = previous
        self._previous.clear()

    def record(self, state):
        identity = id(state)
        if identity not in self._seen:
            self._seen.add(identity)
            self._touched.append(state)

    def restore(self):
        self.active = False
        for state in self._touched:
            value = self._values[id(state)]
            if state._lower_bound != -math.inf or state._upper_bound != math.inf or state._keep_feasible:
                state.value = value
            else:
                state._value = value
        self._touched.clear()
        self._seen.clear()
        for state, value in self._fallback:
            if type(state) is State and state._expr is None and type(value) is float:
                current = state._value
                if current is value or (type(current) is float and current == value and
                                        (value != 0.0 or math.copysign(1.0, current) ==
                                         math.copysign(1.0, value))):
                    continue
            state.value = value
