"""Configuration objects for FullFlow transient solves.

Only timestep-level settings live here.  The nonlinear solve itself reuses the
same :class:`LeastSquaresSettings` object used by the steady-state solver so
bounds, SciPy methods, Jacobian choices, and residual acceptance stay
consistent between steady and transient workflows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransientSettings:
    """Controls the outer transient time-marching loop.

    Parameters
    ----------
    dt : float
        User-selected nominal timestep in seconds.  The solver uses this
        timestep unless it must shorten a step to hit final time, a sequence
        breakpoint, a saved-output time, or a failed-step retry.

    t_final : float
        Final simulation time in seconds.  The solver starts from the current
        ``network.time`` value and advances until this value is reached.

    max_step_retries : int, default=8
        Number of automatic half-step retries allowed after a failed timestep.
        This keeps the public API simple: users can usually call
        ``Transient(network).solve(dt=..., t_final=...)`` without manually
        reducing ``dt`` after the first nonlinear failure.

    minimum_dt : float, optional
        Smallest timestep allowed during automatic retry.  If omitted, the
        solver uses ``dt * 1e-9``.

    Notes
    -----
    This object intentionally does not contain state bounds.  Bounds already
    live on ``State`` objects and are collected into SciPy bounds exactly like
    the steady-state solver does.
    """

    dt: float
    t_final: float
    max_step_retries: int = 8
    minimum_dt: float | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the transient time settings are invalid."""
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive. Got {self.dt}")
        if self.t_final < 0.0:
            raise ValueError(f"t_final must be nonnegative. Got {self.t_final}")
        if self.max_step_retries < 0:
            raise ValueError(
                f"max_step_retries must be nonnegative. Got {self.max_step_retries}"
            )
        if self.minimum_dt is not None and self.minimum_dt <= 0.0:
            raise ValueError(f"minimum_dt must be positive. Got {self.minimum_dt}")

    @property
    def retry_floor(self) -> float:
        """Return the active minimum timestep for automatic retry."""
        if self.minimum_dt is not None:
            return float(self.minimum_dt)

        return float(self.dt) * 1.0e-9
