"""Configuration objects for FullFlow transient solves.

Only timestep-level settings live here.  The nonlinear solve itself reuses the
same :class:`LeastSquaresSettings` object used by the steady-state solver so
bounds, SciPy methods, Jacobian choices, and residual acceptance stay consistent
between steady and transient workflows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransientSettings:
    """Controls the outer transient time-marching loop.

    Parameters
    ----------
    dt : float
        Requested timestep size in seconds.  The first implementation uses a
        fixed timestep.  No adaptive-error control is performed.

    t_final : float
        Final simulation time in seconds.  The solver starts from the current
        ``network.time`` value and advances until this value is reached.

    Notes
    -----
    This object intentionally does not contain state bounds.  Bounds already
    live on ``State`` objects and are collected into SciPy bounds exactly like
    the steady-state solver does.
    """

    dt: float
    t_final: float

    def validate(self) -> None:
        """Raise ``ValueError`` if the transient time settings are invalid."""
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive. Got {self.dt}")
        if self.t_final < 0.0:
            raise ValueError(f"t_final must be nonnegative. Got {self.t_final}")
