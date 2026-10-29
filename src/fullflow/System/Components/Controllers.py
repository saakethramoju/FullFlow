from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class PID(Component):
    """PID controller with output limits and integral anti-windup."""

    TRANSIENT_ONLY = True

    def __init__(
        self,
        name: str,
        network: Network,
        feedback: State,
        setpoint: State | float,
        proportional_gain: float = 0.0,
        integral_gain: float = 0.0,
        derivative_gain: float = 0.0,
        trim: State | float | None = None,
        command: State | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        self.setup()

        self.error = State(0.0)
        self.integral_error = State(0.0)
        self.integral_error_dot = State(0.0)
        self.proportional = State(0.0)
        self.integral = State(0.0)
        self.derivative = State(0.0)
        self.raw_command = State(0.0)
        self.saturated = State(False)
        self.command_bias = State()

        if self.command.is_derived:
            raise ValueError(f"{self.name}: command must be a writable State.")

        if self.minimum.is_assigned and self.maximum.is_assigned:
            if self.minimum.value > self.maximum.value:
                raise ValueError(f"{self.name}: minimum must not exceed maximum.")

        if not self.command.is_assigned and self.trim.is_assigned:
            self.command.value = float(self.trim.value)

    def evaluate_states(self) -> None:
        trim = float(self.trim.value) if self.trim.is_assigned else 0.0
        minimum = float(self.minimum.value) if self.minimum.is_assigned else None
        maximum = float(self.maximum.value) if self.maximum.is_assigned else None

        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{self.name}: minimum must not exceed maximum.")

        if not self.command.is_assigned:
            if not self.trim.is_assigned:
                raise ValueError(f"{self.name}: initialize command or provide trim.")
            self.command.value = trim

        error = float(self.setpoint.value) - float(self.feedback.value)
        kp = float(self.proportional_gain.value)
        ki = float(self.integral_gain.value)
        kd = float(self.derivative_gain.value)
        integral_error = float(self.integral_error.value)

        proportional = kp * error
        integral = ki * integral_error

        if not self.command_bias.is_assigned:
            self.command_bias.value = float(self.command.value) - trim - proportional - integral

        dt = 0.0
        try:
            dt = float(self.network.time.value) - float(self.network.time.previous)
        except ValueError:
            pass

        derivative = 0.0
        if dt > 0.0 and kd != 0.0:
            try:
                derivative = kd * (error - float(self.error.previous)) / dt
            except ValueError:
                pass

        raw_command = trim + float(self.command_bias.value) + proportional + integral + derivative
        command = raw_command
        if minimum is not None:
            command = max(command, minimum)
        if maximum is not None:
            command = min(command, maximum)

        candidate = integral_error
        if dt > 0.0:
            try:
                candidate = float(self.integral_error.previous) + error * dt
            except ValueError:
                pass

        candidate_command = raw_command + ki * (candidate - integral_error)
        direction = ki * error
        hold = (
            maximum is not None and candidate_command > maximum and direction > 0.0
        ) or (
            minimum is not None and candidate_command < minimum and direction < 0.0
        )

        self.error.value = error
        self.proportional.value = proportional
        self.integral.value = integral
        self.derivative.value = derivative
        self.integral_error_dot.value = 0.0 if hold else error
        self.raw_command.value = raw_command
        self.saturated.value = command != raw_command
        self.command.value = command

    @property
    def dynamics(self):
        return [(self.integral_error, self.integral_error_dot)]
