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
        Active timestep size in seconds.  For fixed-step solves this is the
        user's timestep.  For adaptive solves this is the initial timestep.

    t_final : float
        Final simulation time in seconds.  The solver starts from the current
        ``network.time`` value and advances until this value is reached.

    adaptive : bool, default=False
        If ``False``, the solver returns to ``dt`` after automatic retries.  If
        ``True``, accepted-step difficulty is used to choose the next timestep.

    max_step_retries : int, default=8
        Number of automatic half-step retries allowed after a failed timestep.
        This keeps the public API simple: users can usually call
        ``Transient(network).solve(dt=..., t_final=...)`` without manually
        tuning tolerances or reducing ``dt`` after the first failure.

    minimum_dt : float, optional
        Smallest timestep allowed during automatic retry.  If omitted, the
        solver uses ``dt * 1e-9``.

    maximum_dt : float, optional
        Largest timestep allowed by adaptive mode.  This is intentionally an
        internal setting, not a normal user-facing option.

    Notes
    -----
    This object intentionally does not contain state bounds.  Bounds already
    live on ``State`` objects and are collected into SciPy bounds exactly like
    the steady-state solver does.
    """

    dt: float
    t_final: float
    adaptive: bool = False
    max_step_retries: int = 8
    minimum_dt: float | None = None
    maximum_dt: float | None = None

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
        if self.maximum_dt is not None and self.maximum_dt <= 0.0:
            raise ValueError(f"maximum_dt must be positive. Got {self.maximum_dt}")
        if self.maximum_dt is not None and self.maximum_dt < self.retry_floor:
            raise ValueError(
                "maximum_dt must be greater than or equal to the active minimum dt. "
                f"Got maximum_dt={self.maximum_dt}, minimum_dt={self.retry_floor}"
            )

    @property
    def retry_floor(self) -> float:
        """Return the active minimum timestep for automatic retry."""
        if self.minimum_dt is not None:
            return float(self.minimum_dt)

        return float(self.dt) * 1.0e-9

    @property
    def timestep_ceiling(self) -> float:
        """Return the active maximum timestep for adaptive stepping."""
        if self.maximum_dt is not None:
            return float(self.maximum_dt)

        return float(self.dt)
