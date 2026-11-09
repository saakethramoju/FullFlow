"""Capture and restore solver operating points for linearization."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from fullflow.System.State import State, is_state_like
from fullflow.System.Component import Component
from fullflow.Solvers.state_copy import NetworkStateSnapshot, field_names as _field_names


_DIAGNOSTIC_ATTRIBUTES = {
    "_attribute_cache",
    "_output_states",
    "_build_count",
    "_cache_hits",
    "_defer_count",
    "_evaluation_count",
    "_last_error",
    "_reuse_count",
}


def _copy(value, array_cache=None):
    """Return a checkpoint-safe copy, reusing unchanged array snapshots."""
    if isinstance(value, Component):
        return value
    if is_state_like(value):
        return value
    if isinstance(value, (str, int, float, bool, type(None), bytes)):
        return value
    if isinstance(value, np.ndarray):
        if array_cache is not None and value.nbytes >= 4096:
            key = id(value)
            prior = array_cache.get(key)
            if prior is not None and prior[0] is value:
                try:
                    equal = np.array_equal(prior[1], value, equal_nan=True)
                except (TypeError, ValueError):
                    try:
                        equal = np.array_equal(prior[1], value)
                    except (TypeError, ValueError):
                        equal = False
                if equal:
                    return prior[1]
            copied = value.copy()
            array_cache[key] = (value, copied)
            return copied
        return value.copy()
    if isinstance(value, OrderedDict):
        return OrderedDict((_copy(k, array_cache), _copy(v, array_cache)) for k, v in tuple(value.items()))
    if isinstance(value, dict):
        return {_copy(k, array_cache): _copy(v, array_cache) for k, v in tuple(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(_copy(v, array_cache) for v in tuple(value))
    return value


def _same(actual, expected):
    """Compare numerical contents while preserving State and Component identity."""
    if is_state_like(expected) or isinstance(expected, Component):
        return actual is expected
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype:
            return False
        try:
            return np.array_equal(actual, expected, equal_nan=True)
        except TypeError:
            return np.array_equal(actual, expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _same(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(
            _same(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, (set, frozenset)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (float, complex, np.number)) and np.isnan(expected):
        return isinstance(actual, (int, float, complex, np.number)) and bool(np.isnan(actual))
    if isinstance(expected, (str, int, float, bool, type(None), bytes, complex, np.number)):
        return actual == expected
    return actual is expected


class OperatingCheckpoint:
    """Capture accepted State values and mutable component history."""

    def __init__(self, network, cache=None):
        """Initialize the helper and its solver state."""
        self.network = network
        self.version = network.version
        self.time = float(network.time.value)
        # The runtime cache is tied to Network.version. Discover the State
        # graph once per network structure; snapshot its values at every step.
        state_refs = None
        if cache is not None and cache.version == network.version:
            if getattr(cache, "_checkpoint_state_version", None) == network.version:
                state_refs = getattr(cache, "_checkpoint_state_refs", None)
        if state_refs is None:
            self.states = NetworkStateSnapshot.capture(network)
            if cache is not None and cache.version == network.version:
                cache._checkpoint_state_refs = tuple(self.states.states)
                cache._checkpoint_state_version = network.version
        else:
            self.states = NetworkStateSnapshot(list(state_refs))
        self.proxies = []
        if cache is not None:
            known = {id(state) for state in self.states.states}
            for item in cache.snapshot_mutable_states(exclude_state_ids=known):
                if item[0] == "state":
                    _, state, value, previous, second_previous = item
                    if state not in self.states.values:
                        self.states.states.append(state)
                        self.states.values[state] = (value, previous, second_previous)
                else:
                    self.proxies.append(item)
        self.cache = cache
        self.attributes = []
        self._array_cache = None
        if cache is not None:
            if getattr(cache, "_checkpoint_array_cache_version", None) != network.version:
                cache._checkpoint_array_cache = {}
                cache._checkpoint_array_cache_version = network.version
            self._array_cache = cache._checkpoint_array_cache
        self._capture_attributes()
        self.signature = self._signature()
        self.modes = self.mode_signature()

    def _capture_attributes(self):
        """Capture mutable attributes needed to restore the operating point."""
        seen = set()

        def visit(obj, depth):
            """Visit nested values while collecting checkpoint data."""
            # Exact built-in scalars cannot contain restorable attributes.
            # Avoid inspecting them on every accepted timestep.
            if type(obj) in (str, int, float, bool, bytes, type(None), complex):
                return
            if depth > 4 or id(obj) in seen or (is_state_like(obj) and not isinstance(obj, Component)) or obj is self.network:
                return
            seen.add(id(obj))
            if isinstance(obj, (list, tuple, set, frozenset)):
                for item in tuple(obj):
                    visit(item, depth + 1)
                return
            if isinstance(obj, dict):
                for item in tuple(obj.values()):
                    visit(item, depth + 1)
                return
            if depth and getattr(type(obj), '__module__', '').startswith(('numpy', 'scipy', 'fullplot', 'builtins')):
                return
            names = _field_names(obj)
            if not names:
                return
            attrs = {}
            for name in names:
                try:
                    value = getattr(obj, name)
                except Exception:
                    continue
                if value is self.network or (is_state_like(value) and not isinstance(value, Component)):
                    continue
                if isinstance(value, (str, int, float, bool, type(None), bytes, np.ndarray, list, tuple, dict, set, frozenset, Component)):
                    attrs[name] = _copy(value, self._array_cache)
                visit(value, depth + 1)
            self.attributes.append((obj, attrs))

        for component in self.network.component_list:
            visit(component, 0)
        for balance in self.network.balance_list:
            visit(balance, 0)

    def _signature(self):
        """Return the checkpoint signature used for validation."""
        values = []
        for state in self.states.states:
            raw = self.states.values[state][0]
            if isinstance(raw, (str, int, float, bool, type(None), bytes, complex, np.number,
                                np.ndarray, list, tuple, dict, set, frozenset)):
                values.append((state, raw, state._expr, state.bounds))
        return values

    def mode_signature(self):
        """Return the signature of the active discrete operating mode."""
        flags = []
        for state in self.states.states:
            if isinstance(state._value, (bool, str)):
                flags.append((id(state), state._value))
        for obj, attrs in self.attributes:
            for name in attrs:
                value = getattr(obj, name, None)
                if isinstance(value, bool) and (name in ('is_active', 'is_triggered', 'has_aborted') or name.startswith('_active') or name.endswith(('_enabled', '_active'))):
                    flags.append((id(obj), name, value))
        return tuple(flags)

    def validate(self):
        """Validate that the checkpoint still matches the network."""
        if self.network.version != self.version:
            raise ValueError('Network structure changed since the successful solve.')
        if not np.isclose(float(self.network.time.value), self.time, rtol=0, atol=1e-12):
            raise ValueError('Network time changed since the successful solve.')
        for state, expected, expression, bounds in self.signature:
            if not _same(state._value, expected) or state._expr is not expression or state.bounds != bounds:
                label = getattr(state, 'label', 'State')
                raise ValueError(f'{label} changed since the successful solve. Solve again.')
        for obj, attributes in self.attributes:
            for name, expected in attributes.items():
                if name in _DIAGNOSTIC_ATTRIBUTES:
                    continue
                if not _same(getattr(obj, name, None), expected):
                    raise ValueError(f'{type(obj).__name__}.{name} changed since the successful solve. Solve again.')
        if self.mode_signature() != self.modes:
            raise ValueError('A component operating mode changed since the successful solve.')

    def restore(self):
        """Restore the captured network and solver state."""
        if self.network.version != self.version:
            raise ValueError('Cannot restore a checkpoint after the network structure changes.')
        self.states.restore()
        if self.cache is not None and self.proxies:
            self.cache.restore_mutable_states(self.proxies)
        for obj, attrs in reversed(self.attributes):
            frozen = bool(getattr(getattr(type(obj), "__dataclass_params__", None), "frozen", False))
            for name, value in attrs.items():
                if frozen:
                    object.__setattr__(obj, name, _copy(value))
                else:
                    setattr(obj, name, _copy(value))


def discrete_signature(cache):
    """Return a signature for the cache's discrete state."""
    values = tuple(
        (id(state), state._value)
        for state in getattr(cache, "all_state_refs", ())
        if isinstance(state, State) and isinstance(state._value, (bool, str))
    )
    history = tuple(
        (id(entry), getattr(entry, "is_active", None), getattr(entry, "is_triggered", None),
         getattr(entry, "has_aborted", None))
        for component in cache.component_list
        for name in ("_command_entries", "_abort_entries")
        for entry in getattr(component, name, ())
    )
    return values + history
