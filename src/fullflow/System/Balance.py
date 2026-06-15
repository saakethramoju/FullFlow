from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from .State import State

if TYPE_CHECKING:
    from fullflow.System import Network


class Balance:
    """User-defined algebraic solve target."""

    def __init__(
        self,
        name: str,
        network: Network,
        variable: State,
        function: Callable[[], float] | State,
        bounds: tuple[float | None, float | None] | None = None,
        keep_feasible: bool = False,
    ) -> None:
        self.name = name
        self.network = network

        if variable.is_derived:
            raise TypeError("variable cannot be a derived State.")
        self.variable = variable

        if bounds is not None and not self.variable.has_bounds:
            self.variable.set_bounds(bounds, keep_feasible=keep_feasible)

        if isinstance(function, State):
            self._residual = lambda: function.value
            self._residual_source = function
        elif callable(function):
            self._residual = function
            self._residual_source = None
        else:
            raise TypeError("function must be a State or a callable returning float.")

        self.network.add_balance(self)

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float | None, float | None] | None,
    ) -> tuple[float, float]:
        if bounds is None:
            return -math.inf, math.inf
        return State._normalize_bounds(bounds)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.variable.bounds

    @property
    def lower_bound(self) -> float:
        return self.variable.lower_bound

    @property
    def upper_bound(self) -> float:
        return self.variable.upper_bound

    @property
    def has_bounds(self) -> bool:
        return self.variable.has_bounds

    @property
    def keep_feasible(self) -> bool:
        return self.variable.keep_feasible

    @property
    def iteration_variables(self) -> list[State]:
        return [self.variable]

    @property
    def residuals(self) -> list[float]:
        return [float(self._residual())]

    def __str__(self) -> str:
        try:
            value = f"{self.variable.value:.4g}"
        except Exception:
            value = "<uninitialized>"

        lower, upper = self.bounds
        return (
            f"Balance(name={self.name}, variable={value}, "
            f"bounds=({lower:.4g}, {upper:.4g}))"
        )

    def __repr__(self) -> str:
        return str(self)
