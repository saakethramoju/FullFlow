from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real
from typing import Any, Callable, Protocol, runtime_checkable


__all__ = [
    "State",
    "StateLike",
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
            return self._expr()

        if self._value is None:
            raise ValueError(f"State {self._code} has no assigned value.")

        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        if self._expr is not None:
            raise AttributeError("Cannot assign to a derived State.")

        if isinstance(value, Real):
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
