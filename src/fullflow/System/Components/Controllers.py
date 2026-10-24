from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class PID(Component):
    """Transient-only proportional-integral-derivative controller."""

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

        self.error = State(0.0).add_label(
            f"{self.name}:error"
        )

        self.integral_error = State(0.0).add_label(
            f"{self.name}:integral_error"
        )

        self.integral_error_dot = State(0.0).add_label(
            f"{self.name}:integral_error_dot"
        )

        self.proportional = State(0.0).add_label(
            f"{self.name}:proportional"
        )
        self.integral = State(0.0).add_label(
            f"{self.name}:integral"
        )
        self.derivative = State(0.0).add_label(
            f"{self.name}:derivative"
        )

        self.raw_command = State(0.0).add_label(
            f"{self.name}:raw_command"
        )

        self.saturated = State(False).add_label(
            f"{self.name}:saturated"
        )

        self.command_bias = State().add_label(
            f"{self.name}:command_bias"
        )


        if self.command.is_derived:
            raise ValueError(
                f"{self.name}: command must be a writable State, "
                "not a derived State."
            )

        if self.minimum.is_assigned and self.maximum.is_assigned:
            if float(self.minimum.value) > float(self.maximum.value):
                raise ValueError(
                    f"{self.name}: minimum must be less than or equal "
                    "to maximum."
                )

        if not self.command.is_assigned and self.trim.is_assigned:
            self.command.value = float(self.trim.value)

    def evaluate_states(self) -> None:
        """Evaluate the PID command and dynamic integral-error derivative."""


        feedback = float(self.feedback.value)
        setpoint = float(self.setpoint.value)

        proportional_gain = float(self.proportional_gain.value)
        integral_gain = float(self.integral_gain.value)
        derivative_gain = float(self.derivative_gain.value)

        trim = (
            float(self.trim.value)
            if self.trim.is_assigned
            else 0.0
        )

        minimum = (
            float(self.minimum.value)
            if self.minimum.is_assigned
            else None
        )
        maximum = (
            float(self.maximum.value)
            if self.maximum.is_assigned
            else None
        )

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"{self.name}: minimum must be less than or equal "
                "to maximum."
            )

        if not self.command.is_assigned:
            if self.trim.is_assigned:
                self.command.value = trim
            else:
                raise ValueError(
                    f"{self.name}: PID requires either a trim value or "
                    "an initialized command State."
                )


        error = setpoint - feedback

        proportional = proportional_gain * error

        integral = (
            integral_gain
            * float(self.integral_error.value)
        )


        if not self.command_bias.is_assigned:
            self.command_bias.value = (
                float(self.command.value)
                - trim
                - proportional
                - integral
            )


        time = float(self.network.time.value)

        try:
            previous_time = float(self.network.time.previous)
            dt = time - previous_time
        except ValueError:
            dt = 0.0

        if dt > 0.0:
            try:
                previous_error = float(self.error.previous)
            except ValueError:
                previous_error = error

            error_rate = (
                error - previous_error
            ) / dt
        else:
            error_rate = 0.0

        derivative = (
            derivative_gain
            * error_rate
        )


        raw_command = (
            trim
            + float(self.command_bias.value)
            + proportional
            + integral
            + derivative
        )


        command = raw_command
        saturated = False

        if minimum is not None and command < minimum:
            command = minimum
            saturated = True

        if maximum is not None and command > maximum:
            command = maximum
            saturated = True


        candidate_integral_error = float(
            self.integral_error.value
        )

        if dt > 0.0:
            try:
                candidate_integral_error = (
                    float(self.integral_error.previous)
                    + error * dt
                )
            except ValueError:
                pass

        candidate_integral = (
            integral_gain
            * candidate_integral_error
        )

        candidate_raw_command = (
            trim
            + float(self.command_bias.value)
            + proportional
            + candidate_integral
            + derivative
        )

        integral_direction = (
            integral_gain
            * error
        )

        hold_integral = False

        if (
            maximum is not None
            and candidate_raw_command > maximum
            and integral_direction > 0.0
        ):
            hold_integral = True

        if (
            minimum is not None
            and candidate_raw_command < minimum
            and integral_direction < 0.0
        ):
            hold_integral = True

        self.integral_error_dot.value = (
            0.0
            if hold_integral
            else error
        )


        self.error.value = error
        self.proportional.value = proportional
        self.integral.value = integral
        self.derivative.value = derivative

        self.raw_command.value = raw_command
        self.saturated.value = saturated

        self.command.value = command

    @property
    def dynamics(self):
        """Return the PID integral-error dynamic equation."""

        return [
            (
                self.integral_error,
                self.integral_error_dot,
            )
        ]