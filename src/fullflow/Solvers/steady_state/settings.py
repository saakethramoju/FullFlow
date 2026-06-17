"""Small configuration objects used by the steady-state solver.

Keeping settings in one file avoids long argument lists inside the lower-level
solver helpers. The public :class:`SteadyState` API still accepts ordinary
keyword arguments; ``solver.py`` converts those keywords into these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StateEvaluationSettings:
    """Controls fixed-point evaluation of component-derived states.

    ``evaluate_states()`` methods can depend on outputs from other components.
    The solver therefore evaluates them repeatedly until the non-iteration
    states stop changing, or until ``max_passes`` is reached. This is separate
    from the nonlinear least-squares iteration.
    """

    max_passes: int = 20
    tolerance: float = 1e-10

    def validate(self) -> None:
        """Raise ``ValueError`` if the state-evaluation settings are invalid."""
        if self.max_passes <= 0:
            raise ValueError(f"state_max_passes must be positive. Got {self.max_passes}")
        if self.tolerance <= 0.0:
            raise ValueError(f"state_tolerance must be positive. Got {self.tolerance}")


@dataclass(slots=True)
class LeastSquaresSettings:
    """Controls the SciPy ``least_squares`` nonlinear solve.

    ``rtol`` is FullFlow's post-solve residual acceptance tolerance. The other
    tolerances are passed directly to SciPy. Bounds are taken from the iteration
    ``State`` objects, not from this settings object.
    """

    solver_method: str = "trf"
    jacobian_method: str = "3-point"
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    rtol: float = 1e-2

    def validate(self) -> None:
        """Normalize method names and raise ``ValueError`` for invalid options."""
        self.solver_method = self.solver_method.lower()
        self.jacobian_method = self.jacobian_method.lower()

        valid_methods = ("trf", "dogbox", "lm")
        if self.solver_method not in valid_methods:
            raise ValueError(
                f"solver_method must be one of {valid_methods}. Got {self.solver_method!r}."
            )

        valid_jacobians = ("2-point", "3-point")
        if self.jacobian_method not in valid_jacobians:
            raise ValueError(
                f"jacobian_method must be one of {valid_jacobians}. Got {self.jacobian_method!r}."
            )

        if self.rtol <= 0.0:
            raise ValueError(f"Residual tolerance (rtol) must be positive. Got {self.rtol}")
