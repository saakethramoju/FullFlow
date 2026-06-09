from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Union, Callable
from .State import State

if TYPE_CHECKING:
    from fullflow.System import Network


class Balance:
    """
    User-defined algebraic solve target.

    `Balance` adds one extra solve equation to a `Network` by telling the solver
    to adjust one assignable `State` until a residual expression evaluates to
    zero.

    A balance contributes one iteration variable and one residual equation to
    the global steady-state solve.

    Parameters
    ----------
    name : str
        Balance name
    network : Network
        Network that owns this balance
    variable : State
        Non-derived state that the solver may modify
    function : callable or State
        Residual expression to drive to zero
    bounds : tuple[float or None, float or None], optional
        Fallback bounds for the solve variable
    keep_feasible : bool, optional
        Whether bounded solvers should keep the variable inside bounds

    Iteration Variables
    -------------------
    variable : State
        State adjusted by the solver

    Residuals
    ---------
    residual : float
        User-defined residual driven to zero by the solver

        ``function = 0``

    Notes
    -----
    The residual equation is automatically included in the network solve.
    During each iteration, the solver modifies `variable` until `function`
    approaches zero.

    The `function` argument may be a callable returning a float:

        ``Balance("Target Pc", network, variable=A, function=lambda: Pc.value - target)``

    or a derived `State` expression:

        ``Balance("Target Pc", network, variable=A, function=Pc - target)``

    Balance-provided bounds are only a fallback. If the `State` already has
    bounds, the `State` bounds take priority.

    Derived states cannot be used as solve variables because they do not store
    assignable values.

    A `State` cannot be solved by both a component residual equation and a
    `Balance`. The owning `Network` checks for overlapping iteration variables.
    """
    def __init__(
        self,
        name: str,
        network: Network,
        variable: State,
        function: Union[Callable[[], float], State],
        bounds: tuple[float | None, float | None] | None = None,
        keep_feasible: bool = False,
    ):
        self.name = name
        self.network = network

        if variable.is_derived:
            raise TypeError("variable cannot be a derived State.")
        else:
            self.variable = variable

        # Balance-provided bounds are only a fallback.
        # If the State already has bounds, the State takes priority.
        if bounds is not None and not self.variable.has_bounds:
            lower, upper = self._normalize_bounds(bounds)
            self.variable._lower_bound = lower
            self.variable._upper_bound = upper
            self.variable._keep_feasible = bool(keep_feasible)

        if isinstance(function, State):
            # Use State math -> evaluate via .value
            self._residual = lambda: function.value
            self._residual_source = function
        elif callable(function):
            self._residual = function
            self._residual_source = None
        else:
            raise TypeError(
                "residual_function must be a State or a callable returning float."
            )

        self.network.add_balance(balance=self)

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float | None, float | None] | None,
    ) -> tuple[float, float]:
        if bounds is None:
            return -np.inf, np.inf

        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                "bounds must be None or a tuple of the form (lower, upper)."
            )

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


    # -------------- STEADY-STATE -------------- #
    
    @property
    def iteration_variables(self) -> list[State]:
        return [self.variable]

    @property
    def residuals(self) -> list[float]:
        return [float(self._residual())]

    # -------------- PRINTING-------------- #
    def __str__(self) -> str:
        try:
            val = f"{self.variable.value:.4g}"
        except Exception:
            val = "<uninitialized>"

        lb, ub = self.bounds

        return (
            f"Balance(name={self.name}, "
            f"variable={val}, "
            f"bounds=({lb:.4g}, {ub:.4g}))"
        )


    def __repr__(self) -> str:
        return self.__str__()