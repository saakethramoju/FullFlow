"""Shared value-copy helpers for restorable solver state."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from types import ModuleType, FunctionType, MethodType
from typing import Any

from fullflow.System.State import State, is_state_like
from fullflow.System.Component import Component


def snapshot_triplet(value, previous, second_previous, copy_value):
    """Copy distinct objects while preserving aliases between time levels."""
    current = copy_value(value)
    prior = current if previous is value else copy_value(previous)
    if second_previous is value:
        older = current
    elif second_previous is previous:
        older = prior
    else:
        older = copy_value(second_previous)
    return current, prior, older


class NetworkStateSnapshot:
    """Capture and restore State values before independent transient runs."""

    @staticmethod
    def _snapshot_value(value: Any) -> Any:
        """Return a restorable copy of a network value."""
        if isinstance(value, Component):
            return value
        if is_state_like(value):
            return value
        if type(value) in (float, int, bool, str, bytes, type(None), complex):
            return value

        if isinstance(value, dict):
            return {
                NetworkStateSnapshot._snapshot_value(key): NetworkStateSnapshot._snapshot_value(item)
                for key, item in tuple(value.items())
            }

        if isinstance(value, list):
            return [NetworkStateSnapshot._snapshot_value(item) for item in tuple(value)]

        if isinstance(value, tuple):
            return tuple(NetworkStateSnapshot._snapshot_value(item) for item in value)

        if isinstance(value, set):
            return {NetworkStateSnapshot._snapshot_value(item) for item in tuple(value)}

        if isinstance(value, frozenset):
            return frozenset(
                NetworkStateSnapshot._snapshot_value(item)
                for item in tuple(value)
            )

        try:
            return deepcopy(value)
        except Exception:
            return value

    def __init__(self, states: list[State]) -> None:
        """Initialize the object and register any FullFlow state wiring."""
        self.states = states
        self.values = {
            state: snapshot_triplet(
                state._value, state._previous, state._second_previous,
                self._snapshot_value,
            )
            for state in states
        }

    @classmethod
    def capture(cls, network) -> "NetworkStateSnapshot":
        """Capture a restorable snapshot of the network state."""
        states: list[State] = []
        seen_objects: set[int] = set()
        seen_states: set[int] = set()

        def visit(value: Any) -> None:
            """Visit nested values while collecting checkpoint data."""
            value_id = id(value)
            if value_id in seen_objects:
                return
            seen_objects.add(value_id)

            if isinstance(value, State):
                if value_id not in seen_states:
                    seen_states.add(value_id)
                    states.append(value)
                if value._expr is None:
                    visit(value._value)
                return

            if isinstance(value, (type, ModuleType, FunctionType, MethodType)):
                return

            if isinstance(value, dict):
                for item in tuple(value.values()):
                    visit(item)
                return

            if isinstance(value, (list, tuple, set, frozenset)):
                for item in tuple(value):
                    visit(item)
                return

            if hasattr(value, "__dict__"):
                for item in tuple(vars(value).values()):
                    visit(item)
            if is_dataclass(value):
                for entry in fields(value):
                    try:
                        item = getattr(value, entry.name)
                    except (AttributeError, TypeError):
                        continue
                    visit(item)

        visit(network)
        return cls(states)

    def restore(self) -> None:
        """Restore the captured network and solver state."""
        for state, (value, previous, second_previous) in self.values.items():
            state._value, state._previous, state._second_previous = snapshot_triplet(
                value, previous, second_previous, self._snapshot_value,
            )

