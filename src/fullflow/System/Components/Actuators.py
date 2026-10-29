from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Actuator(Component):
    """Command-to-position actuator with optional limits and slew rate."""

    def __init__(
        self,
        name: str,
        network: Network,
        command: State | float,
        position: State | float | None = None,
        minimum: State | float | None = None,
        maximum: State | float | None = None,
        rate_limit: State | float | None = None,
    ) -> None:
        self.setup()
        self.velocity = State(0.0)
        self.position_error = State(0.0)
        self.raw_command = State(0.0)
        self.limited_command = State(0.0)
        self.rate_limited = State(False)
        self.saturated = State(False)

    def evaluate_states(self) -> None:
        if not self.command.is_assigned:
            raise ValueError(f"{self.name}: command is uninitialized.")

        raw = float(self.command.value)
        minimum = float(self.minimum.value) if self.minimum.is_assigned else None
        maximum = float(self.maximum.value) if self.maximum.is_assigned else None

        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{self.name}: minimum must not exceed maximum.")

        command = raw
        if minimum is not None:
            command = max(command, minimum)
        if maximum is not None:
            command = min(command, maximum)

        initial = float(self.position.value) if self.position.is_assigned else command
        position = initial
        if minimum is not None:
            position = max(position, minimum)
        if maximum is not None:
            position = min(position, maximum)
        saturated = raw != command or position != initial

        previous = position
        dt = 0.0
        rate_limited = False
        if self.rate_limit.is_assigned:
            try:
                dt = float(self.network.time.value) - float(self.network.time.previous)
                previous = float(self.position.previous)
            except ValueError:
                dt = 0.0

            if dt > 0.0:
                step = abs(float(self.rate_limit.value)) * dt
                rate_limited = abs(command - previous) > step
                position = max(previous - step, min(command, previous + step))
        else:
            position = command

        unclamped = position
        if minimum is not None:
            position = max(position, minimum)
        if maximum is not None:
            position = min(position, maximum)

        self.position.value = position
        self.velocity.value = (position - previous) / dt if dt > 0.0 else 0.0
        self.position_error.value = command - position
        self.raw_command.value = raw
        self.limited_command.value = command
        self.rate_limited.value = rate_limited
        self.saturated.value = saturated or position != unclamped
