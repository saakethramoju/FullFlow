import math
import numpy as np


class State:
    """
    Value container used throughout FullFlow.

    `State` can store numeric values, object values, or derived values computed
    from other states. Arithmetic and solver math require numeric states.

    Use `.value` for the stored object/value.
    Use `.numeric_value` when a float is required.
    """

    def __init__(
        self,
        value=None,
        *,
        bounds: tuple[float | None, float | None] | None = None,
        keep_feasible: bool = False,
    ):
        self._expr = None
        self._lower_bound, self._upper_bound = self._normalize_bounds(bounds)
        self._keep_feasible = bool(keep_feasible)
        self._value = None
        self._code = hex(id(self))

        if value is not None:
            self.value = value

    @classmethod
    def _derived(cls, expr):
        state = cls()
        state._expr = expr
        return state

    @staticmethod
    def _normalize_bounds(bounds):
        if bounds is None:
            return -np.inf, np.inf

        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("bounds must be None or a tuple of the form (lower, upper).")

        lower, upper = bounds
        lower = -np.inf if lower is None else float(lower)
        upper = np.inf if upper is None else float(upper)

        if lower > upper:
            raise ValueError(
                f"Invalid bounds: lower bound {lower} is greater than upper bound {upper}."
            )

        return lower, upper

    def _validate_bounds(self, v: float) -> None:
        if v < self._lower_bound:
            raise ValueError(f"Value {v} is below the lower bound of {self._lower_bound}.")
        if v > self._upper_bound:
            raise ValueError(f"Value {v} is above the upper bound of {self._upper_bound}.")

    @property
    def value(self):
        if self._expr is not None:
            return self._expr()

        if self._value is None:
            raise ValueError(f"State {self._code} has no assigned value.")

        return self._value

    @value.setter
    def value(self, v) -> None:
        if self._expr is not None:
            raise AttributeError("Cannot assign to a derived State.")

        if isinstance(v, (int, float, np.number)):
            v = float(v)
            self._validate_bounds(v)

        self._value = v

    @property
    def numeric_value(self) -> float:
        value = self.value

        try:
            value = float(value)
        except Exception as exc:
            raise TypeError(
                f"State {self._code} contains non-numeric value "
                f"{type(value).__name__!r} and cannot be used in numeric math."
            ) from exc

        self._validate_bounds(value)
        return value

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
        return not (self._lower_bound == -np.inf and self._upper_bound == np.inf)

    @property
    def keep_feasible(self) -> bool:
        return self._keep_feasible

    def is_within_bounds(self, v=None) -> bool:
        if v is None:
            v = self.numeric_value
        else:
            v = float(v)

        return self._lower_bound <= v <= self._upper_bound

    def _value_string_for_display(self) -> str:
        if self.is_derived:
            try:
                return f"{self.value} <derived>"
            except Exception:
                return "<derived>"

        if self._value is None:
            return "<uninitialized>"

        return str(self._value)

    def __str__(self) -> str:
        value_str = self._value_string_for_display()

        if self.has_bounds:
            return f"State(code={self._code}, value={value_str}, bounds={self.bounds})"

        return f"State(code={self._code}, value={value_str})"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _coerce(other) -> "State":
        if isinstance(other, State):
            return other
        return State(other)

    # ---------- arithmetic ----------
    def __add__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.numeric_value + other.numeric_value)

    def __radd__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.numeric_value + self.numeric_value)

    def __sub__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.numeric_value - other.numeric_value)

    def __rsub__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.numeric_value - self.numeric_value)

    def __mul__(self, other):
        try:
            other = self._coerce(other)
            return State._derived(lambda: self.numeric_value * other.numeric_value)
        except (TypeError, ValueError):
            if hasattr(other, "__rmul__"):
                return other.__rmul__(self)
            raise

    def __rmul__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.numeric_value * self.numeric_value)

    def __truediv__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.numeric_value / other.numeric_value)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.numeric_value / self.numeric_value)

    def __pow__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.numeric_value ** other.numeric_value)

    def __rpow__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.numeric_value ** self.numeric_value)

    def __neg__(self):
        return State._derived(lambda: -self.numeric_value)

    def __abs__(self):
        return State._derived(lambda: abs(self.numeric_value))

    # ---------- basic math ----------
    def sqrt(self):
        return State._derived(lambda: math.sqrt(self.numeric_value))

    def exp(self):
        return State._derived(lambda: math.exp(self.numeric_value))

    def expm1(self):
        return State._derived(lambda: math.expm1(self.numeric_value))

    def log(self):
        return State._derived(lambda: math.log(self.numeric_value))

    def log10(self):
        return State._derived(lambda: math.log10(self.numeric_value))

    def log2(self):
        return State._derived(lambda: math.log2(self.numeric_value))

    def log1p(self):
        return State._derived(lambda: math.log1p(self.numeric_value))

    # ---------- trig ----------
    def sin(self):
        return State._derived(lambda: math.sin(self.numeric_value))

    def cos(self):
        return State._derived(lambda: math.cos(self.numeric_value))

    def tan(self):
        return State._derived(lambda: math.tan(self.numeric_value))

    def asin(self):
        return State._derived(lambda: math.asin(self.numeric_value))

    def acos(self):
        return State._derived(lambda: math.acos(self.numeric_value))

    def atan(self):
        return State._derived(lambda: math.atan(self.numeric_value))

    # ---------- hyperbolic ----------
    def sinh(self):
        return State._derived(lambda: math.sinh(self.numeric_value))

    def cosh(self):
        return State._derived(lambda: math.cosh(self.numeric_value))

    def tanh(self):
        return State._derived(lambda: math.tanh(self.numeric_value))

    def asinh(self):
        return State._derived(lambda: math.asinh(self.numeric_value))

    def acosh(self):
        return State._derived(lambda: math.acosh(self.numeric_value))

    def atanh(self):
        return State._derived(lambda: math.atanh(self.numeric_value))

    # ---------- angle conversions ----------
    def degrees(self):
        return State._derived(lambda: math.degrees(self.numeric_value))

    def radians(self):
        return State._derived(lambda: math.radians(self.numeric_value))

    # ---------- rounding / integer ----------
    def floor(self):
        return State._derived(lambda: math.floor(self.numeric_value))

    def ceil(self):
        return State._derived(lambda: math.ceil(self.numeric_value))

    def trunc(self):
        return State._derived(lambda: math.trunc(self.numeric_value))

    @staticmethod
    def maximum(a, b):
        a = State._coerce(a)
        b = State._coerce(b)
        return State._derived(lambda: max(a.numeric_value, b.numeric_value))

    @staticmethod
    def minimum(a, b):
        a = State._coerce(a)
        b = State._coerce(b)
        return State._derived(lambda: min(a.numeric_value, b.numeric_value))

    def clip(self, lower=None, upper=None):
        result = self

        if lower is not None:
            result = State.maximum(result, lower)

        if upper is not None:
            result = State.minimum(result, upper)

        return result

    # ---------- misc ----------
    def modf(self):
        return (
            State._derived(lambda: math.modf(self.numeric_value)[0]),
            State._derived(lambda: math.modf(self.numeric_value)[1]),
        )

    def fmod(self, other):
        other = self._coerce(other)
        return State._derived(lambda: math.fmod(self.numeric_value, other.numeric_value))

    def hypot(self, other):
        other = self._coerce(other)
        return State._derived(lambda: math.hypot(self.numeric_value, other.numeric_value))

    def copysign(self, other):
        other = self._coerce(other)
        return State._derived(lambda: math.copysign(self.numeric_value, other.numeric_value))

    def __format__(self, format_spec: str) -> str:
        return format(self.numeric_value, format_spec)