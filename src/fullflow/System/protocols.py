from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol, runtime_checkable


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
    """Cached State-like type test that avoids probing object instances.

    FullFlow contains dynamic proxy objects such as CallableLookupAttribute.
    Calling ``hasattr(proxy, ...)`` can accidentally create a proxy attribute or
    evaluate a lookup.  This helper inspects class dictionaries only, and caches
    by type so solver hot paths do not repeatedly walk MROs.
    """
    if bool(getattr(value_type, "_fullflow_state_like", False)):
        return True

    return any(
        "value" in getattr(cls, "__dict__", {})
        and "is_assigned" in getattr(cls, "__dict__", {})
        for cls in value_type.__mro__
    )


def is_state_like(value: Any) -> bool:
    """Return True for State and State-compatible FullFlow proxy objects."""
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
    """Return True for State-like objects the solver may assign.

    Derived ``State`` expressions are readable but not assignable.
    ``CallableLookupAttribute`` objects report ``is_derived`` for display
    compatibility, but they are assignable proxy inputs/guesses and are marked
    at the class level.
    """
    if not is_state_like(value):
        return False

    if bool(getattr(type(value), "_fullflow_assignable_state_like", False)):
        return True

    try:
        return not bool(value.is_derived)
    except Exception:
        return True
