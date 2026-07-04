from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network





class PID(Component):
    """Simple PID controller.

    The controller reads a feedback value and setpoint, then writes a command.

        error = setpoint - feedback

        command = trim
                + proportional_gain * error
                + integral_gain * integral(error)
                + derivative_gain * d(error)/dt

    trim is the nominal command. The PID terms are corrections around trim.
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
        trim: State | float = 0.0,
        command: State | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        # Runtime memory is created on the first evaluation.
        #
        # setup() already converted constructor inputs into State-like
        # attributes, so command/minimum/maximum are State objects even when
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

            # If no command State was supplied, setup() made an unassigned
            # State. Initialize that output to zero.
            if not self.command.is_assigned:
                self.command.value = 0.0

        # Current simulation time.
        time = float(self.network.time.value)

        # Read controller inputs.
        feedback = float(self.feedback.value)
        setpoint = float(self.setpoint.value)
        trim = float(self.trim.value)

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
        raw_command = trim + proportional + integral + derivative
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