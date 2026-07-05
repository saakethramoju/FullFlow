from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Actuator(Component):
    """Simple actuator with optional limits and optional rate limiting.

    The actuator reads a commanded value and writes an actual position.

        command  -> requested actuator value
        position -> actual actuator value used by the plant

    If rate_limit is omitted, the actuator is ideal:

        position = command

    If rate_limit is supplied, the actuator moves toward command no faster than:

        abs(position_dot) <= rate_limit

    minimum and maximum are optional hard limits.

    The constructor intentionally only calls setup(). All initialization is done
    during evaluation so the component follows the normal FullFlow component
    pattern.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        command: State | float,
        position: State | float | None = None,
        minimum: State | float | None = None,
        maximum: State | float | None = None,
        rate_limit: State | float | None = None,
    ):
        self.setup()

    def _limits(self):
        minimum = None
        maximum = None

        if self.minimum.is_assigned:
            minimum = float(self.minimum.value)

        if self.maximum.is_assigned:
            maximum = float(self.maximum.value)

        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"{self.name}: minimum must be less than or equal to maximum."
            )

        return minimum, maximum

    def _clamp(self, value: float):
        minimum, maximum = self._limits()

        saturated = False

        if minimum is not None and value < minimum:
            value = minimum
            saturated = True

        if maximum is not None and value > maximum:
            value = maximum
            saturated = True

        return value, saturated

    def _initialize(self):
        self._initialized = True

        self.previous_time = State(float(self.network.time.value))

        self.velocity = State(0.0)
        self.position_error = State(0.0)
        self.raw_command = State(0.0)
        self.limited_command = State(0.0)
        self.rate_limited = State(False)
        self.saturated = State(False)

        if not self.command.is_assigned:
            raise ValueError(
                f"{self.name}: Actuator requires an initialized command."
            )

        raw_command = float(self.command.value)
        limited_command, saturated = self._clamp(raw_command)

        self.raw_command.value = raw_command
        self.limited_command.value = limited_command

        if not self.position.is_assigned:
            self.position.value = limited_command
        else:
            position, position_saturated = self._clamp(float(self.position.value))
            self.position.value = position
            saturated = saturated or position_saturated

        self.position_error.value = limited_command - float(self.position.value)
        self.saturated.value = saturated

    def evaluate_states(self):
        if not hasattr(self, "_initialized"):
            self._initialize()

        if not self.command.is_assigned:
            raise ValueError(
                f"{self.name}: Actuator command is uninitialized."
            )

        time = float(self.network.time.value)
        dt = time - float(self.previous_time.value)

        raw_command = float(self.command.value)
        command, command_saturated = self._clamp(raw_command)

        position = float(self.position.value)

        self.raw_command.value = raw_command
        self.limited_command.value = command

        rate_limited = False
        saturated = command_saturated

        if self.rate_limit.is_assigned:
            if dt > 0.0:
                rate_limit = abs(float(self.rate_limit.value))
                max_step = rate_limit * dt

                delta = command - position

                if delta > max_step:
                    delta = max_step
                    rate_limited = True

                if delta < -max_step:
                    delta = -max_step
                    rate_limited = True

                next_position = position + delta
                next_position, position_saturated = self._clamp(next_position)

                self.position.value = next_position
                self.velocity.value = (next_position - position) / dt
                self.previous_time.value = time

                saturated = saturated or position_saturated

            else:
                next_position = position
                self.velocity.value = 0.0

        else:
            next_position, position_saturated = self._clamp(command)

            self.position.value = next_position
            self.velocity.value = 0.0

            if dt > 0.0:
                self.previous_time.value = time

            saturated = saturated or position_saturated

        self.position_error.value = command - float(self.position.value)
        self.rate_limited.value = rate_limited
        self.saturated.value = saturated