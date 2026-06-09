import math
import numpy as np


class State:
    """
    Scalar value container used throughout FullFlow.

    `State` objects store variables that are shared between components,
    balances, and solvers. A State may hold a directly assigned value or
    represent a derived value computed from other states.

    States support arithmetic operations, allowing derived states to be
    constructed naturally:

        ``density_average = 0.5 * (density1 + density2)``

    Placeholder states may be created with:

        ``pressure = State()``

    and assigned later by a component or solver.

    Parameters
    ----------
    value : float, optional
        Initial state value.
    bounds : tuple[float or None, float or None], optional
        Lower and upper bounds used by bounded solvers.
    keep_feasible : bool, optional
        Indicates that bounded solvers should attempt to keep the state
        within its bounds during iterations.

    Notes
    -----
    Derived states are read-only and cannot be assigned directly.

    Accessing `.value` on an uninitialized non-derived state raises
    `ValueError`.
    """
    def __init__(
        self,
        value: float | None = None,
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
            value = float(value)
            self._validate_bounds(value)
            self._value = value

    @classmethod
    def _derived(cls, expr):
        state = cls()
        state._expr = expr
        return state

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float | None, float | None] | None,
    ) -> tuple[float, float]:
        if bounds is None:
            return -np.inf, np.inf

        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("bounds must be None or a tuple of the form (lower, upper).")

        lower, upper = bounds

        if lower is None:
            lower = -np.inf
        else:
            lower = float(lower)

        if upper is None:
            upper = np.inf
        else:
            upper = float(upper)

        if lower > upper:
            raise ValueError(
                f"Invalid bounds: lower bound {lower} is greater than upper bound {upper}."
            )

        return lower, upper

    def _validate_bounds(self, v: float) -> None:
        if v < self._lower_bound:
            raise ValueError(
                f"Value {v} is below the lower bound of {self._lower_bound}."
            )
        if v > self._upper_bound:
            raise ValueError(
                f"Value {v} is above the upper bound of {self._upper_bound}."
            )

    @property
    def value(self) -> float:
        """
        Return the current numeric value.

        Raises
        ------
        ValueError
            If this is a non-derived State that has not been assigned yet.
        """
        if self._expr is not None:
            return self._expr()

        if self._value is None:
            raise ValueError(
                f"State {self._code} has no assigned value.\n\n"

                "This State was created as a placeholder using State() and has "
                "not yet been assigned a numeric value.\n\n"

                "A placeholder State() is valid if some component, solver iteration, "
                "or external assignment sets the value before the State is accessed.\n\n"

                "This error occurs only when `.value` is accessed before assignment.\n"
                "Typical scenarios include:\n"
                "  - A derived expression depends on an uninitialized State\n"
                "      Example:\n"
                "          rho_avg = 0.5 * (rho1 + rho2)\n"
                "      If rho1.value or rho2.value is requested before assignment,\n"
                "      evaluating rho_avg.value will fail.\n\n"

                "  - A component evaluates a State before another component computes it\n"
                "      Example:\n"
                "          A line model requests density.value before a volume model\n"
                "          has computed and assigned the density.\n\n"

                "  - The solver evaluates the residual function before an iteration\n"
                "      variable has received an initial value.\n\n"

                "Note:\n"
                "  Placeholder States are allowed and often useful.\n"
                "  For example, mass flow iteration variables may safely start as\n"
                "  State() if the solver assigns them before they are evaluated.\n\n"

                "Fixes:\n"
                "  - Move a state evaluation/computation to the corresponding\n"
                "    component's 'pre_evaluation()'\n\n"

                "  - Provide an initial value:\n"
                "        density = State(800)\n"
                "        mdot = State(0)\n\n"

                "  - Ensure the State is assigned before any component or expression\n"
                "    accesses `.value`\n\n"

                "  - Ensure producer components evaluate before consumer components"
            )

        return self._value

    @value.setter
    def value(self, v: float) -> None:
        if self._expr is not None:
            raise AttributeError("Cannot assign to a derived State.")

        v = float(v)
        self._validate_bounds(v)
        self._value = v

    @property
    def is_derived(self) -> bool:
        return self._expr is not None

    @property
    def is_assigned(self) -> bool:
        return self._expr is not None or self._value is not None

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._lower_bound, self._upper_bound)

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

    def is_within_bounds(self, v: float | None = None) -> bool:
        if v is None:
            v = self.value

        if v < self._lower_bound:
            return False
        if v > self._upper_bound:
            return False
        return True

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
        value_str = self._value_string_for_display()

        if self.has_bounds:
            return f"State(code={self._code}, value={value_str}, bounds={self.bounds})"

        return f"State(code={self._code}, value={value_str})"

    @staticmethod
    def _coerce(other) -> "State":
        if isinstance(other, State):
            return other
        return State(float(other))

    # ---------- arithmetic ----------
    def __add__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.value + other.value)

    def __radd__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.value + self.value)

    def __sub__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.value - other.value)

    def __rsub__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.value - self.value)

    def __mul__(self, other):
        try:
            other = self._coerce(other)
            return State._derived(lambda: self.value * other.value)
        except (TypeError, ValueError):
            if hasattr(other, "__rmul__"):
                return other.__rmul__(self)
            raise

    def __rmul__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.value * self.value)

    def __truediv__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.value / other.value)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.value / self.value)

    def __pow__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: self.value ** other.value)

    def __rpow__(self, other):
        other = self._coerce(other)
        return State._derived(lambda: other.value ** self.value)

    def __neg__(self):
        return State._derived(lambda: -self.value)

    def __abs__(self):
        return State._derived(lambda: abs(self.value))

    # ---------- basic math ----------
    def sqrt(self):
        return State._derived(lambda: math.sqrt(self.value))

    def exp(self):
        return State._derived(lambda: math.exp(self.value))

    def expm1(self):
        return State._derived(lambda: math.expm1(self.value))

    def log(self):
        return State._derived(lambda: math.log(self.value))

    def log10(self):
        return State._derived(lambda: math.log10(self.value))

    def log2(self):
        return State._derived(lambda: math.log2(self.value))

    def log1p(self):
        return State._derived(lambda: math.log1p(self.value))

    # ---------- trig ----------
    def sin(self):
        return State._derived(lambda: math.sin(self.value))

    def cos(self):
        return State._derived(lambda: math.cos(self.value))

    def tan(self):
        return State._derived(lambda: math.tan(self.value))

    def asin(self):
        return State._derived(lambda: math.asin(self.value))

    def acos(self):
        return State._derived(lambda: math.acos(self.value))

    def atan(self):
        return State._derived(lambda: math.atan(self.value))

    # ---------- hyperbolic ----------
    def sinh(self):
        return State._derived(lambda: math.sinh(self.value))

    def cosh(self):
        return State._derived(lambda: math.cosh(self.value))

    def tanh(self):
        return State._derived(lambda: math.tanh(self.value))

    def asinh(self):
        return State._derived(lambda: math.asinh(self.value))

    def acosh(self):
        return State._derived(lambda: math.acosh(self.value))

    def atanh(self):
        return State._derived(lambda: math.atanh(self.value))

    # ---------- angle conversions ----------
    def degrees(self):
        return State._derived(lambda: math.degrees(self.value))

    def radians(self):
        return State._derived(lambda: math.radians(self.value))

    # ---------- rounding / integer ----------
    def floor(self):
        return State._derived(lambda: math.floor(self.value))

    def ceil(self):
        return State._derived(lambda: math.ceil(self.value))

    def trunc(self):
        return State._derived(lambda: math.trunc(self.value))

    @staticmethod
    def maximum(a, b):
        a = State._coerce(a)
        b = State._coerce(b)
        return State._derived(lambda: max(a.value, b.value))

    @staticmethod
    def minimum(a, b):
        a = State._coerce(a)
        b = State._coerce(b)
        return State._derived(lambda: min(a.value, b.value))

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
            State._derived(lambda: math.modf(self.value)[0]),
            State._derived(lambda: math.modf(self.value)[1]),
        )

    def fmod(self, other):
        other = self._coerce(other)
        return State._derived(lambda: math.fmod(self.value, other.value))

    def hypot(self, other):
        other = self._coerce(other)
        return State._derived(lambda: math.hypot(self.value, other.value))

    def copysign(self, other):
        other = self._coerce(other)
        return State._derived(lambda: math.copysign(self.value, other.value))

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)