from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Union, Callable
from .State import State

if TYPE_CHECKING:
    from fullflow.System import Network


class Balance:
    """
    Adds an additional solve equation to the network by defining:

        "adjust this variable until this function equals zero"

    A Balance contributes:
        - one iteration variable
        - one residual equation

    to the global steady-state solve.

    Parameters
    ----------
    variable
        Non-derived State that the solver is allowed to modify.

    function
        Residual expression to drive to zero. May be:
            - a callable returning a float
            - a derived State expression

    bounds
        Optional fallback bounds for the solve variable.
        Only applied if the State itself does not already
        define bounds.

    keep_feasible
        Passed through to bounded solvers such as
        scipy.optimize.least_squares.

    Examples
    --------
    Balance chamber pressure by adjusting injector area:

        Balance(
            "Balance injector for Pc",
            network=net,
            variable=Injector.A,
            function=Chamber.Pc - 300 * PSIA_TO_PA,
        )

    Balance thrust by adjusting nozzle efficiency:

        Balance(
            "Balance thrust",
            network=net,
            variable=Nozzle.eta_Cf,
            function=Nozzle.F - 200 * N_TO_LBF,
            bounds=(0, 1),
        )

    Notes
    -----
    The residual equation is automatically included in the
    network solve. During each iteration, the solver modifies
    `variable` until `function -> 0`.

    Derived States cannot be used as solve variables because
    they do not store assignable values.
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