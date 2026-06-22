from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real
from typing import Any, Callable, Protocol, runtime_checkable
from collections.abc import Iterable, Mapping


__all__ = [
    "State",
    "is_state_like",
    "resolve_value",
    "resolve_numeric",
    "is_assignable_state_like",
]


@runtime_checkable
class StateLike(Protocol):
    """Runtime shape used by solvers for assignable scalar-like quantities."""

    @property
    def value(self) -> Any: ...

    @value.setter
    def value(self, value: Any) -> None: ...

    @property
    def is_assigned(self) -> bool: ...

    @property
    def numeric_value(self) -> float: ...

    @property
    def lower_bound(self) -> float: ...

    @property
    def upper_bound(self) -> float: ...

    @property
    def has_bounds(self) -> bool: ...

    @property
    def keep_feasible(self) -> bool: ...


@lru_cache(maxsize=512)
def _is_state_like_type(value_type: type) -> bool:
    if bool(getattr(value_type, "_fullflow_state_like", False)):
        return True

    return any(
        "value" in getattr(cls, "__dict__", {})
        and "is_assigned" in getattr(cls, "__dict__", {})
        for cls in value_type.__mro__
    )


def is_state_like(value: Any) -> bool:
    """Return True for State and State-compatible proxy objects."""
    return _is_state_like_type(type(value))


def resolve_value(value: Any) -> Any:
    """Return the underlying value for State-like inputs; otherwise unchanged."""
    return value.value if is_state_like(value) else value


def resolve_numeric(value: Any) -> float:
    """Return a float from a State-like or plain numeric value."""
    if is_state_like(value):
        return float(value.numeric_value)
    return float(value)


def is_assignable_state_like(value: Any) -> bool:
    """Return True for State-like objects the solver may assign."""
    if not is_state_like(value):
        return False

    if bool(getattr(type(value), "_fullflow_assignable_state_like", False)):
        return True

    try:
        return not bool(value.is_derived)
    except Exception:
        return True


class State:
    """Lightweight value container used throughout FullFlow."""

    _fullflow_state_like = True

    __slots__ = (
        "_value",
        "_expr",
        "_previous",
        "_second_previous",
        "_discrete_frozen",
        "_proposed_value",
        "_lower_bound",
        "_upper_bound",
        "_keep_feasible",
        "_code",
    )

    def __init__(
        self,
        value: Any = None,
        *,
        bounds: tuple[float | None, float | None] | None = None,
        keep_feasible: bool = False,
    ) -> None:
        self._value: Any = None
        self._expr: Callable[[], Any] | None = None
        self._previous: Any = None
        self._second_previous: Any = None
        self._discrete_frozen = False
        self._proposed_value: Any = None
        self._lower_bound, self._upper_bound = self._normalize_bounds(bounds)
        self._keep_feasible = bool(keep_feasible)
        self._code = hex(id(self))

        if value is not None:
            self.value = value

    @classmethod
    def _derived(cls, expr: Callable[[], Any]) -> "State":
        state = cls()
        state._expr = expr
        return state
    
    @classmethod
    def from_iterable(cls, value: Any) -> Any:
        """
        Recursively convert values inside an iterable/container into States.

        This is useful when a dictionary, list, tuple, set, or other iterable
        stores values that should become solver variables.

        Existing scalar State-like values are preserved. State-like containers
        are unwrapped first, then their contained values are converted.

        Strings and bytes are treated as scalar values, not iterated
        character-by-character.

        Examples
        --------
        Convert a normal dictionary:

            State.from_iterable({"H2": 0.2, "O2": 0.8})

        gives:

            {
                "H2": State(0.2),
                "O2": State(0.8),
            }

        Convert a LookupAttribute whose value is a dictionary:

            State.from_iterable(ChamberEquilibrium.composition)

        also gives a dictionary of States.

        Notes
        -----
        This helper does not normalize fractions, enforce sum-to-one, or add
        N - 1 composition closure. It only wraps contained scalar values in
        State objects.
        """

        def is_container(item: Any) -> bool:
            if isinstance(item, (str, bytes, bytearray)):
                return False

            if isinstance(item, Mapping):
                return True

            if isinstance(item, (list, tuple, set, frozenset)):
                return True

            return isinstance(item, Iterable)

        if is_state_like(value):
            try:
                resolved = value.value
            except Exception:
                return value

            if is_container(resolved):
                return cls.from_iterable(resolved)

            return value

        if isinstance(value, Mapping):
            return {
                key: cls.from_iterable(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                cls.from_iterable(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                cls.from_iterable(item)
                for item in value
            )

        if isinstance(value, set):
            return {
                cls.from_iterable(item)
                for item in value
            }

        if isinstance(value, frozenset):
            return frozenset(
                cls.from_iterable(item)
                for item in value
            )

        if isinstance(value, (str, bytes, bytearray)):
            return cls(value)

        if isinstance(value, Iterable):
            return [
                cls.from_iterable(item)
                for item in value
            ]

        return cls(value)

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float | None, float | None] | None,
    ) -> tuple[float, float]:
        if bounds is None:
            return -math.inf, math.inf

        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("bounds must be None or a (lower, upper) tuple.")

        lower, upper = bounds
        lower = -math.inf if lower is None else float(lower)
        upper = math.inf if upper is None else float(upper)

        if lower > upper:
            raise ValueError(
                f"Invalid bounds: lower bound {lower} is greater than upper bound {upper}."
            )

        return lower, upper

    def _validate_bounds(self, value: float) -> None:
        if value < self._lower_bound:
            raise ValueError(
                f"Value {value} is below the lower bound of {self._lower_bound}."
            )
        if value > self._upper_bound:
            raise ValueError(
                f"Value {value} is above the upper bound of {self._upper_bound}."
            )

    def _requires_numeric_value(self) -> bool:
        return self.has_bounds or self.keep_feasible

    def _as_numeric(self, value: Any) -> float:
        try:
            numeric_value = float(value)
        except Exception as exc:
            raise TypeError(
                f"State {self._code} contains non-numeric value "
                f"{type(value).__name__!r} and cannot be used in numeric math."
            ) from exc

        return numeric_value

    @property
    def value(self) -> Any:
        if self._expr is not None:
            try:
                return self._expr()
            except Exception:
                if self._value is not None:
                    return self._value
                raise

        if self._value is None:
            raise ValueError(f"State {self._code} has no assigned value.")

        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        if self._expr is not None:
            raise AttributeError("Cannot assign to a derived State.")

        if isinstance(value, bool):
            pass
        elif isinstance(value, Real):
            value = float(value)
            self._validate_bounds(value)
        elif self._requires_numeric_value():
            value = self._as_numeric(value)
            self._validate_bounds(value)

        self._value = value

    def set(self, value: Any) -> "State":
        """Assign a value and return this State for fluent setup."""
        self.value = value
        return self

    def derive_from(self, source: Any) -> "State":
        """Make this State derive its value from another State-like object or callable.

        This mutates the existing State object, so components that already
        reference this State keep referencing the same object. If this State
        already has a stored value, that value is kept as a startup fallback
        until the derived expression can be evaluated.
        """
        if source is self:
            raise ValueError("A State cannot derive from itself.")

        if is_state_like(source):
            self._expr = lambda: source.value
        elif callable(source):
            self._expr = source
        else:
            source = State(source)
            self._expr = lambda: source.value

        return self

    def __ilshift__(self, source: Any) -> "State":
        """In-place shorthand for ``derive_from``.

        Example
        -------
        ``mixture_ratio <<= ox_mdot / fuel_mdot``
        """
        return self.derive_from(source)

    @property
    def previous(self) -> Any:
        """Value from the most recently accepted transient step."""
        if self._previous is None:
            raise ValueError(
                f"State {self._code} has no previous. "
                "Transient history has not been initialized."
            )

        return self._previous

    @property
    def second_previous(self) -> Any:
        """Value from one accepted transient step before ``previous``."""
        if self._second_previous is None:
            raise ValueError(
                f"State {self._code} has no second_previous. "
                "Transient history has not been initialized for two accepted steps."
            )

        return self._second_previous

    def store_previous(self) -> None:
        """Store the current value as transient history.

        The transient solver calls this after each accepted timestep. Components
        should not need to call it directly.
        """
        if self.is_assigned:
            self._second_previous = self._previous
            self._previous = self.value

    def clear_previous(self) -> None:
        """Clear stored transient history."""
        self._previous = None
        self._second_previous = None

    def propose(
        self,
        value: Any,
        *,
        turn_on: float | None = None,
        turn_off: float | None = None,
    ) -> bool:
        """Safely propose a boolean/discrete mode value.

        Components use this for discontinuous branches such as cavitation,
        choking, limiters, or bang-bang valves.  In normal evaluation the
        proposed value is assigned immediately.  During a transient nonlinear
        solve, the transient solver freezes discrete States; proposals are then
        remembered but the currently accepted value is returned so the residual
        equations do not jump branches while SciPy is perturbing variables.

        The simple form uses a boolean condition::

            is_open = self.is_open.propose(pressure > opening_pressure)

        The threshold form adds hysteresis::

            is_open = self.is_open.propose(
                pressure,
                turn_on=opening_pressure,
                turn_off=closing_pressure,
            )

        With hysteresis, a False state turns True only when ``value > turn_on``;
        a True state turns False only when ``value < turn_off``.  Between those
        thresholds, the current mode is kept.
        """
        if (turn_on is None) != (turn_off is None):
            raise ValueError(
                "turn_on and turn_off must either both be provided or both be omitted."
            )

        if turn_on is None:
            proposed_value = bool(value)
        else:
            turn_on_value = float(turn_on)
            turn_off_value = float(turn_off)

            if turn_on_value < turn_off_value:
                raise ValueError("turn_on must be greater than or equal to turn_off.")

            signal = float(value)
            current_value = bool(self.value)

            if current_value:
                proposed_value = signal > turn_off_value
            else:
                proposed_value = signal > turn_on_value

        if self._discrete_frozen:
            self._proposed_value = proposed_value
            return bool(self.value)

        self.value = proposed_value
        self._proposed_value = proposed_value
        return bool(self.value)

    def freeze_discrete(self) -> None:
        """Freeze this State for safe discrete-mode proposals.

        The transient solver calls this before a timestep nonlinear solve.  It
        has no effect on normal numeric assignment; it only changes the behavior
        of :meth:`propose`.
        """
        self._discrete_frozen = True
        self._proposed_value = None

    def commit_discrete(self) -> None:
        """Accept the last proposed discrete value and unfreeze this State."""
        proposed_value = self._proposed_value
        self._discrete_frozen = False
        self._proposed_value = None

        if proposed_value is not None:
            self.value = proposed_value

    def reject_discrete(self) -> None:
        """Discard any proposed discrete value and unfreeze this State."""
        self._discrete_frozen = False
        self._proposed_value = None

    @property
    def numeric_value(self) -> float:
        value = self._as_numeric(self.value)
        self._validate_bounds(value)
        return value

    @property
    def code(self) -> str:
        return self._code

    @property
    def is_numeric(self) -> bool:
        try:
            self.numeric_value
            return True
        except Exception:
            return False

    @property
    def is_derived(self) -> bool:
        return self._expr is not None

    @property
    def is_assigned(self) -> bool:
        return self._expr is not None or self._value is not None

    @property
    def bounds(self) -> tuple[float, float]:
        return self._lower_bound, self._upper_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def has_bounds(self) -> bool:
        return self._lower_bound != -math.inf or self._upper_bound != math.inf

    @property
    def keep_feasible(self) -> bool:
        return self._keep_feasible

    def is_within_bounds(self, value: Any = None) -> bool:
        numeric_value = self.numeric_value if value is None else self._as_numeric(value)
        return self._lower_bound <= numeric_value <= self._upper_bound

    def set_bounds(
        self,
        bounds: tuple[float | None, float | None] | None,
        *,
        keep_feasible: bool | None = None,
    ) -> "State":
        """Update bounds in place and return ``self`` for chaining."""
        self._lower_bound, self._upper_bound = self._normalize_bounds(bounds)
        if keep_feasible is not None:
            self._keep_feasible = bool(keep_feasible)
        if self._value is not None and isinstance(self._value, Real):
            self._validate_bounds(float(self._value))
        return self

    def _value_string_for_display(self) -> str:
        if self.is_derived:
            try:
                return f"{self.value} <derived>"
            except Exception:
                return "<derived>"

        if self._value is None:
            return "<uninitialized>"

        return str(self._value)

    def __float__(self) -> float:
        return self.numeric_value

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __iter__(self):
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, item: Any) -> bool:
        return item in self.value

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        current = self.value
        current[key] = value
        self.value = current

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        try:
            value = self.value
        except Exception as exc:
            raise AttributeError(name) from exc

        return getattr(value, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value(*args, **kwargs)

    def __array__(self, dtype=None):
        import numpy as np

        return np.asarray(self.value, dtype=dtype)

    def __format__(self, format_spec: str) -> str:
        return format(self.numeric_value, format_spec)

    def __str__(self) -> str:
        value = self._value_string_for_display()
        if self.has_bounds:
            return f"State(code={self._code}, value={value}, bounds={self.bounds})"
        return f"State(code={self._code}, value={value})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def _coerce(other: Any) -> "State":
        return other if isinstance(other, State) else State(other)

    def _binary(self, other: Any, op: Callable[[float, float], float]) -> "State":
        other = self._coerce(other)
        return State._derived(lambda: op(self.numeric_value, other.numeric_value))

    def _rbinary(self, other: Any, op: Callable[[float, float], float]) -> "State":
        other = self._coerce(other)
        return State._derived(lambda: op(other.numeric_value, self.numeric_value))

    def __add__(self, other: Any) -> "State":
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "State":
        return self._rbinary(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> "State":
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "State":
        return self._rbinary(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> "State":
        try:
            return self._binary(other, lambda a, b: a * b)
        except (TypeError, ValueError):
            if hasattr(other, "__rmul__"):
                return other.__rmul__(self)
            raise

    def __rmul__(self, other: Any) -> "State":
        return self._rbinary(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> "State":
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> "State":
        return self._rbinary(other, lambda a, b: a / b)

    def __pow__(self, other: Any) -> "State":
        return self._binary(other, lambda a, b: a**b)

    def __rpow__(self, other: Any) -> "State":
        return self._rbinary(other, lambda a, b: a**b)

    def __neg__(self) -> "State":
        return State._derived(lambda: -self.numeric_value)

    def __abs__(self) -> "State":
        return State._derived(lambda: abs(self.numeric_value))
            
    def _compare(self, other: Any, op: Callable[[float, float], bool]) -> bool:
        return op(self.numeric_value, resolve_numeric(other))

    def __lt__(self, other: Any) -> bool:
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other: Any) -> bool:
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other: Any) -> bool:
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other: Any) -> bool:
        return self._compare(other, lambda a, b: a >= b)

    def _unary(self, func: Callable[[float], float]) -> "State":
        return State._derived(lambda: func(self.numeric_value))

    def sqrt(self) -> "State": return self._unary(math.sqrt)
    def exp(self) -> "State": return self._unary(math.exp)
    def expm1(self) -> "State": return self._unary(math.expm1)
    def log(self) -> "State": return self._unary(math.log)
    def log10(self) -> "State": return self._unary(math.log10)
    def log2(self) -> "State": return self._unary(math.log2)
    def log1p(self) -> "State": return self._unary(math.log1p)
    def sin(self) -> "State": return self._unary(math.sin)
    def cos(self) -> "State": return self._unary(math.cos)
    def tan(self) -> "State": return self._unary(math.tan)
    def asin(self) -> "State": return self._unary(math.asin)
    def acos(self) -> "State": return self._unary(math.acos)
    def atan(self) -> "State": return self._unary(math.atan)
    def sinh(self) -> "State": return self._unary(math.sinh)
    def cosh(self) -> "State": return self._unary(math.cosh)
    def tanh(self) -> "State": return self._unary(math.tanh)
    def asinh(self) -> "State": return self._unary(math.asinh)
    def acosh(self) -> "State": return self._unary(math.acosh)
    def atanh(self) -> "State": return self._unary(math.atanh)
    def degrees(self) -> "State": return self._unary(math.degrees)
    def radians(self) -> "State": return self._unary(math.radians)
    def floor(self) -> "State": return self._unary(math.floor)
    def ceil(self) -> "State": return self._unary(math.ceil)
    def trunc(self) -> "State": return self._unary(math.trunc)

    @staticmethod
    def maximum(a: Any, b: Any) -> "State":
        a = State._coerce(a)
        b = State._coerce(b)
        return State._derived(lambda: max(a.numeric_value, b.numeric_value))

    @staticmethod
    def minimum(a: Any, b: Any) -> "State":
        a = State._coerce(a)
        b = State._coerce(b)
        return State._derived(lambda: min(a.numeric_value, b.numeric_value))

    def clip(self, lower: Any = None, upper: Any = None) -> "State":
        result = self
        if lower is not None:
            result = State.maximum(result, lower)
        if upper is not None:
            result = State.minimum(result, upper)
        return result

    def modf(self) -> tuple["State", "State"]:
        return (
            State._derived(lambda: math.modf(self.numeric_value)[0]),
            State._derived(lambda: math.modf(self.numeric_value)[1]),
        )

    def fmod(self, other: Any) -> "State":
        return self._binary(other, math.fmod)

    def hypot(self, other: Any) -> "State":
        return self._binary(other, math.hypot)

    def copysign(self, other: Any) -> "State":
        return self._binary(other, math.copysign)
