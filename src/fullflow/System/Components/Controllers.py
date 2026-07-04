from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network



class PID(Component):
    """Simple PID controller.

    The controller reads a feedback value and setpoint, then writes a command.

        error = setpoint - feedback

        command = command_bias
                + proportional_gain * error
                + integral_gain * integral(error)
                + derivative_gain * d(error)/dt

    The command bias comes from trim when trim is supplied.

    If trim is omitted but an initialized command State is supplied, the initial
    command value is used as the fixed command bias. This lets a user connect the
    PID to an actuator without also providing a trim value.

    If both trim and an initialized command are omitted, the PID cannot know what
    actuator command to start from, so it raises an error.
    """

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
    ):
        self.setup()

    def evaluate_states(self):
        # Runtime memory is created on the first evaluation.
        #
        # setup() already converted constructor inputs into State-like
        # attributes, so trim/command/minimum/maximum are State objects even when
        # their values are None.
        if not hasattr(self, "_initialized"):
            self._initialized = True

            # Controller memory.
            self.integral_error = State(0.0)
            self.previous_error = State(0.0)
            self.previous_time = State(float(self.network.time.value))

            # Public diagnostic outputs.
            self.error = State(0.0)
            self.proportional = State(0.0)
            self.integral = State(0.0)
            self.derivative = State(0.0)
            self.raw_command = State(0.0)
            self.saturated = State(False)

            # If trim was supplied, trim is the nominal actuator command.
            #
            # If trim was not supplied but command already has a value, use the
            # initial command value as the fixed nominal actuator command. This
            # avoids inventing a universal PID default like 0.0.
            #
            # If neither trim nor command is assigned, the PID has no way to know
            # what actuator units or nominal command it should use.
            if self.trim.is_assigned:
                self.command_bias = State(float(self.trim.value))

            elif self.command.is_assigned:
                self.command_bias = State(float(self.command.value))

            else:
                raise ValueError(
                    f"{self.name}: PID requires either a trim value or an "
                    "initialized command State. The PID cannot choose a default "
                    "actuator command because command units depend on what is "
                    "being controlled."
                )

            # If the PID owns the command State, initialize it to the bias.
            if not self.command.is_assigned:
                self.command.value = float(self.command_bias.value)

        # Current simulation time.
        time = float(self.network.time.value)

        # Read controller inputs.
        feedback = float(self.feedback.value)
        setpoint = float(self.setpoint.value)

        # If trim is supplied, it remains the live nominal command. This allows
        # a user to schedule or otherwise update trim. If trim was omitted, the
        # initial command value captured above is used as the fixed bias.
        if self.trim.is_assigned:
            command_bias = float(self.trim.value)
        else:
            command_bias = float(self.command_bias.value)

        # Read gains.
        proportional_gain = float(self.proportional_gain.value)
        integral_gain = float(self.integral_gain.value)
        derivative_gain = float(self.derivative_gain.value)

        # Main control error.
        error = setpoint - feedback

        # Time step since the controller last advanced.
        #
        # If the nonlinear solver evaluates the component multiple times at the
        # same network time, dt is zero, so the integral does not grow again.
        dt = time - float(self.previous_time.value)

        # Candidate integral and derivative.
        #
        # The integral candidate is only committed after saturation/anti-windup
        # logic is checked.
        if dt > 0.0:
            next_integral_error = float(self.integral_error.value) + error * dt
            error_rate = (error - float(self.previous_error.value)) / dt
        else:
            next_integral_error = float(self.integral_error.value)
            error_rate = 0.0

        # PID terms.
        proportional = proportional_gain * error
        integral = integral_gain * next_integral_error
        derivative = derivative_gain * error_rate

        # Unclamped command.
        raw_command = command_bias + proportional + integral + derivative
        command = raw_command

        # Clamp command to optional limits.
        saturated = False

        minimum = self.minimum.value if self.minimum.is_assigned else None
        maximum = self.maximum.value if self.maximum.is_assigned else None

        if minimum is not None and command < minimum:
            command = minimum
            saturated = True

        if maximum is not None and command > maximum:
            command = maximum
            saturated = True

        # Simple always-on anti-windup.
        #
        # If saturated and the error would push the command farther into the
        # saturated limit, do not commit the new integral value.
        commit_integral = True

        if saturated:
            if maximum is not None and command >= maximum and error > 0.0:
                commit_integral = False

            if minimum is not None and command <= minimum and error < 0.0:
                commit_integral = False

        # Write diagnostic outputs.
        self.error.value = error
        self.proportional.value = proportional
        self.integral.value = integral
        self.derivative.value = derivative
        self.raw_command.value = raw_command
        self.saturated.value = saturated

        # Write command output.
        self.command.value = command

        # Advance controller memory only when time advances.
        if dt > 0.0:
            if commit_integral:
                self.integral_error.value = next_integral_error

            self.previous_error.value = error
            self.previous_time.value = time