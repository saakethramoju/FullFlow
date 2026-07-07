from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Actuator(Component):
    """Command-to-position actuator with optional saturation and rate limiting.

        The actuator reads ``command`` and writes ``position``.  If ``rate_limit`` is
        omitted, position follows command exactly after applying optional minimum and
        maximum limits.  If ``rate_limit`` is supplied, position can only move by
        ``rate_limit * dt`` per transient step.  Diagnostic states record raw command,
        limited command, position error, velocity, saturation, and whether rate
        limiting was active.

        Use this component between controllers/sequences and physical plant states
        when valve or actuator dynamics should be represented without writing a new
        valve component."""

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
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
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
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
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
