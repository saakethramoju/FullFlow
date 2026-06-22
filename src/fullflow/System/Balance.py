from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .State import State, is_assignable_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Balance:
    def __init__(
        self,
        name: str,
        network: Network,
        variable: State,
        function: Callable[[], State | float] | State,
        bounds: tuple[float | None, float | None] | None = None,
        keep_feasible: bool = False,
    ) -> None:
        if not is_assignable_state_like(variable):
            raise TypeError("variable must be an assignable, non-derived State.")

        self.name = name
        self.network = network
        self.variable = variable

        if bounds is not None and not variable.has_bounds:
            variable.set_bounds(bounds, keep_feasible=keep_feasible)

        if isinstance(function, State):
            self._residual = lambda: function.value
        elif callable(function):
            self._residual = function
        else:
            raise TypeError("function must be a State or a callable returning a State or float.")

        network.add_balance(self)

    @property
    def iteration_variables(self) -> list[State]:
        return [self.variable]

    @property
    def residuals(self) -> list[State | float]:
        return [self._residual()]

    def __str__(self) -> str:
        try:
            value = f"{self.variable.value:.4g}"
        except Exception:
            value = "<uninitialized>"

        lower, upper = self.variable.bounds
        return (
            f"Balance(name={self.name}, variable={value}, "
            f"bounds=({lower:.4g}, {upper:.4g}))"
        )

    def __repr__(self) -> str:
        return str(self)