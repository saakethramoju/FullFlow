from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass

import numpy as np

from fullflow.System.State import State, is_state_like
from fullflow.Solvers.transient.models import NetworkStateSnapshot


_DIAGNOSTIC_ATTRIBUTES = {
    "_build_count",
    "_cache_hits",
    "_defer_count",
    "_evaluation_count",
    "_reuse_count",
}


def _copy(value):
    if is_state_like(value):
        return value
    if isinstance(value, (str, int, float, bool, type(None), bytes)):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, OrderedDict):
        return OrderedDict((_copy(k), _copy(v)) for k, v in value.items())
    if isinstance(value, dict):
        return {_copy(k): _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(_copy(v) for v in value)
    return value


def _field_names(obj):
    if is_dataclass(obj):
        return tuple(f.name for f in fields(obj))
    return tuple(vars(obj)) if hasattr(obj, '__dict__') else ()


class OperatingCheckpoint:
    """Capture accepted State values and mutable component history."""

    def __init__(self, network, cache=None):
        self.network = network
        self.version = network.version
        self.models = tuple((m.name, m.active_option_name) for m in network.model_list)
        self.time = float(network.time.value)
        self.states = NetworkStateSnapshot.capture(network)
        self.proxies = cache.snapshot_mutable_states() if cache is not None else None
        self.cache = cache
        self.attributes = []
        self._capture_attributes()
        self.signature = self._signature()
        self.modes = self.mode_signature()

    def _capture_attributes(self):
        seen = set()

        def visit(obj, depth):
            if depth > 4 or id(obj) in seen or is_state_like(obj) or obj is self.network:
                return
            seen.add(id(obj))
            if isinstance(obj, (list, tuple, set, frozenset)):
                for item in obj:
                    visit(item, depth + 1)
                return
            if isinstance(obj, dict):
                for item in obj.values():
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
                if value is self.network or is_state_like(value):
                    continue
                if isinstance(value, (str, int, float, bool, type(None), bytes, np.ndarray, list, tuple, dict, set, frozenset)):
                    attrs[name] = _copy(value)
                visit(value, depth + 1)
            self.attributes.append((obj, attrs))

        for component in self.network.component_list:
            visit(component, 0)
        for balance in self.network.balance_list:
            visit(balance, 0)

    def _signature(self):
        values = []
        for state in self.states.states:
            raw = state._value
            if isinstance(raw, (str, int, float, bool, type(None))):
                values.append((state, raw, state._expr, state.bounds))
            elif isinstance(raw, np.ndarray):
                values.append((state, raw.copy(), state._expr, state.bounds))
        return values

    def mode_signature(self):
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
        if self.network.version != self.version or tuple((m.name, m.active_option_name) for m in self.network.model_list) != self.models:
            raise ValueError('Network structure or model option changed since the successful solve.')
        if not np.isclose(float(self.network.time.value), self.time, rtol=0, atol=1e-12):
            raise ValueError('Network time changed since the successful solve.')
        for state, expected, expression, bounds in self.signature:
            actual = state._value
            if isinstance(expected, np.ndarray):
                try:
                    matched = isinstance(actual, np.ndarray) and np.array_equal(actual, expected, equal_nan=True)
                except TypeError:
                    matched = isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
            elif isinstance(expected, float) and np.isnan(expected):
                matched = isinstance(actual, (int, float)) and np.isnan(actual)
            else:
                matched = actual == expected
            if not matched or state._expr is not expression or state.bounds != bounds:
                label = getattr(state, 'label', 'State')
                raise ValueError(f'{label} changed since the successful solve. Solve again.')
        for obj, attributes in self.attributes:
            for name, expected in attributes.items():
                if name in _DIAGNOSTIC_ATTRIBUTES:
                    continue
                if isinstance(expected, (int, float, str, bool, type(None))):
                    current = getattr(obj, name, None)
                    if isinstance(expected, float) and np.isnan(expected):
                        same = isinstance(current, (int, float)) and np.isnan(current)
                    else:
                        same = current == expected
                    if not same:
                        raise ValueError(f'{type(obj).__name__}.{name} changed since the successful solve. Solve again.')
        if self.mode_signature() != self.modes:
            raise ValueError('A component operating mode changed since the successful solve.')

    def restore(self):
        if self.network.version != self.version:
            raise ValueError('Cannot restore a checkpoint after the network structure changes.')
        self.states.restore()
        if self.proxies is not None:
            self.cache.restore_mutable_states(self.proxies)
        for obj, attrs in reversed(self.attributes):
            for name, value in attrs.items():
                setattr(obj, name, _copy(value))


def discrete_signature(cache):
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
