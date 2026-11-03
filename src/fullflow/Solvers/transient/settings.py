"""Configuration objects for FullFlow transient solves."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class TransientSettings:
    """Controls the outer transient time-marching loop."""

    dt: float
    t_final: float
    max_step_retries: int = 8
    minimum_dt: float | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the transient time settings are invalid."""
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be finite and positive. Got {self.dt}")
        if not math.isfinite(self.t_final) or self.t_final < 0.0:
            raise ValueError(f"t_final must be finite and nonnegative. Got {self.t_final}")
        if self.max_step_retries < 0:
            raise ValueError(
                f"max_step_retries must be nonnegative. Got {self.max_step_retries}"
            )
        if self.minimum_dt is not None and (not math.isfinite(self.minimum_dt) or self.minimum_dt <= 0.0):
            raise ValueError(f"minimum_dt must be finite and positive. Got {self.minimum_dt}")

    @property
    def retry_floor(self) -> float:
        """Return the active minimum timestep for automatic retry."""
        if self.minimum_dt is not None:
            return float(self.minimum_dt)

        return float(self.dt) * 1.0e-9
