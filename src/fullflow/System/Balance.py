from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .State import State, is_assignable_state_like, label_state_refs

if TYPE_CHECKING:
    from fullflow.System import Network


class Balance:
    """User-defined algebraic residual attached directly to a network.

        ``Balance`` is the lightest way to tell the steady-state or transient solver
        to vary one state until an externally computed residual is zero.  Component
        classes should normally expose their own ``balances`` property, but a user
        balance is convenient for one-off closure equations, calibration variables,
        map inversions, or coupling equations that do not deserve a new component.

        Parameters
        ----------
        variable : State-like
            Iteration variable adjusted by the solver.  The variable's bounds are
            forwarded to SciPy.
        residual : State-like or float
            Residual value to drive to zero.  It may be a derived ``State`` or any
            state-like object updated by component evaluation.

        Notes
        -----
        User balances can be ignored with ``ignore_balances`` in solver calls.  They
        are exported separately from component residuals so debugging tables can
        identify which equation supplied each residual."""
    def __init__(
        self,
        name: str,
        network: Network,
        variable: State,
        function: Callable[[], State | float] | State,
        bounds: tuple[float | None, float | None] | None = None,
        keep_feasible: bool = False,
    ) -> None:
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        if not is_assignable_state_like(variable):
            raise TypeError("variable must be an assignable, non-derived State.")

        self.name = name
        self.network = network
        self.variable = variable
        label_state_refs(self.variable, f"{self.name}:variable")

        if bounds is not None and not variable.has_bounds:
            variable.set_bounds(bounds, keep_feasible=keep_feasible)

        if isinstance(function, State):
            label_state_refs(function, f"{self.name}:residual")
            self._residual = lambda: function.value
        elif callable(function):
            self._residual = function
        else:
            raise TypeError("function must be a State or a callable returning a State or float.")

        network.add_balance(self)

    @property
    def balances(self) -> list[tuple[State, State | float]]:
        """Algebraic equation exposed to the solvers.

        ``Balance`` is the user-facing way to add a plain algebraic closure:

            variable -> adjusted by the solver
            residual -> driven to zero

        Components use the same ``balances`` convention.
        """
        return [(self.variable, self._residual)]

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
