"""Live, fixed-shape groups of scalar State references."""

import numpy as np

from .State import State
from fullflow.Exceptions import UnassignedStateError

__all__ = ["StateArray"]


class StateArray:
    """Group existing scalar States without introducing solver variables."""

    _fullflow_state_like = True
    _fullflow_state_array = True
    __array_priority__ = 1000

    def __init__(self, members):
        def unpack(value):
            if isinstance(value, State):
                return (), [value]
            if isinstance(value, StateArray):
                return value.shape, list(value._members.flat)
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise TypeError("StateArray members must be scalar States.")
            children = [unpack(child) for child in value]
            shape = children[0][0] if children else ()
            if any(child_shape != shape for child_shape, _ in children):
                raise ValueError("StateArray cannot contain ragged shapes.")
            return (len(children),) + shape, [s for _, states in children for s in states]

        shape, states = unpack(members)
        self._members = np.empty(shape, dtype=object)
        for index, state in zip(np.ndindex(shape), states):
            self._members[index] = state
        self._members.flags.writeable = False
        try:
            self.value
        except UnassignedStateError:
            pass

    @classmethod
    def full(cls, shape, value):
        array = np.empty(shape, dtype=object)
        if np.ndim(value) != 0 or np.asarray(value).dtype.kind not in 'biuf':
            raise TypeError("StateArray.full requires a numeric scalar value.")
        for index in np.ndindex(array.shape):
            array[index] = State(value)
        return cls(array)

    @property
    def shape(self):
        return self._members.shape

    @property
    def size(self):
        return self._members.size

    @property
    def value(self):
        values = []
        for state in self._members.flat:
            value = state.value
            if np.ndim(value) != 0 or np.asarray(value).dtype.kind not in 'biuf':
                raise TypeError("StateArray members must contain numeric scalar values.")
            values.append(value)
        return np.asarray(values).reshape(self.shape)

    @value.setter
    def value(self, value):
        values = np.asarray(value)
        if values.shape != self.shape:
            raise ValueError(f"StateArray shape must be {self.shape}; got {values.shape}.")
        if values.dtype.kind not in 'biuf':
            raise TypeError("StateArray requires numeric values.")
        assignments = {}
        for state, number in zip(self._members.flat, values.flat):
            if state.is_derived:
                raise AttributeError("Cannot assign to derived StateArray members.")
            state._validate_bounds(float(number))
            previous = assignments.get(state)
            if previous is not None and previous != number:
                raise ValueError("Repeated StateArray members require identical assigned values.")
            assignments[state] = number
        for state, number in assignments.items():
            state.value = number

    @property
    def is_assigned(self):
        return all(state.is_assigned for state in self._members.flat)

    @property
    def is_derived(self):
        return any(state.is_derived for state in self._members.flat)

    def at(self, index):
        selected = self._members[index]
        return selected if isinstance(selected, State) else StateArray(selected)

    def __getitem__(self, index):
        return self.at(index)

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return (self.at(i) for i in range(len(self)))

    def __array__(self, dtype=None, copy=None):
        return np.array(self.value, dtype=dtype, copy=True)

    def _math_value(self):
        return self.value

    def _expression(self):
        return State._derived(lambda: self.value)

    def sum(self, axis=None):
        return self._expression().sum(axis=axis)

    def mean(self, axis=None):
        return self._expression().mean(axis=axis)

    def __add__(self, other): return self._expression() + other
    def __radd__(self, other): return other + self._expression()
    def __sub__(self, other): return self._expression() - other
    def __rsub__(self, other): return other - self._expression()
    def __mul__(self, other): return self._expression() * other
    def __rmul__(self, other): return other * self._expression()
    def __truediv__(self, other): return self._expression() / other
    def __rtruediv__(self, other): return other / self._expression()
    def __neg__(self): return -self._expression()
