from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class PID(Component):
    """Transient-only proportional-integral-derivative controller.

    The controller calculates the feedback error as:

        error = setpoint - feedback

    and calculates the requested command as:

        raw_command = trim
                    + command_bias
                    + proportional
                    + integral
                    + derivative

    where:

        proportional = proportional_gain * error

        integral = integral_gain * integral_error

        derivative = derivative_gain * d(error)/dt

    The final command is the raw command limited by ``minimum`` and
    ``maximum`` when those limits are supplied.

    Bumpless startup
    ----------------
    A FullFlow model will commonly obtain a steady-state operating point before
    beginning a transient simulation. Because this PID is transient-only, it
    does not overwrite the command during the steady-state solve.

    When transient control begins, ``command_bias`` is initialized so that the
    first PID output equals the command value already established by the user or
    steady-state model. This provides bumpless transfer whether or not ``trim``
    is supplied.

    The initial bias is:

        command_bias = initial_command
                     - initial_trim
                     - initial_proportional
                     - initial_integral

    The initial derivative contribution is taken as zero because no accepted
    transient error history is yet available.

    Trim
    ----
    ``trim`` is treated as a live feed-forward contribution. Its value is
    included directly in the command during every transient evaluation.

    Supplying trim therefore does not disable bumpless startup. The initial
    command still determines the first controller output, while later changes
    in trim produce immediate feed-forward changes in the requested command.

    Integral dynamics
    -----------------
    The accumulated error is represented as a normal FullFlow dynamic State:

        d(integral_error)/dt = error

    FullFlow integrates this State together with the rest of the transient
    network. The PID does not manually advance or commit integral history during
    ``evaluate_states()``.

    This is important because ``evaluate_states()`` may be called many times
    while the nonlinear solver converges one timestep. Accepted transient
    history is obtained through the public ``State.previous`` interface.

    Anti-windup
    -----------
    Conditional integration is used to prevent the integral term from driving
    the controller farther into saturation.

    Integration is suspended only when:

    1. the predicted command lies beyond a command limit; and
    2. the integral contribution is moving the command farther beyond that
       limit.

    Integration remains active when the integral contribution would move the
    command back toward the valid range. The direction test includes the sign
    of ``integral_gain``, so the logic also works for negative controller gains.

    Derivative behavior
    -------------------
    The derivative term acts on error rather than directly on feedback.
    Therefore, an abrupt setpoint change can produce derivative kick when
    ``derivative_gain`` is nonzero.

    Notes
    -----
    This component is intentionally marked ``TRANSIENT_ONLY``. During a
    steady-state solve, the PID does not vary its command or force its control
    error to zero. The user or steady-state model establishes the initial
    operating point, and the PID takes control when transient integration
    begins.

    Physical actuator limits such as valve opening rate, motor acceleration, or
    command slew rate should generally be represented with an ``Actuator`` or
    another physical component. The PID command limits only clamp the requested
    command value.

    Parameters
    ----------
    name
        User-facing component name.
    network
        FullFlow network containing the controller.
    feedback
        Measured or simulated process value.
    setpoint
        Desired process value.
    proportional_gain
        Proportional gain applied to the current error.
    integral_gain
        Integral gain applied to accumulated error.
    derivative_gain
        Derivative gain applied to the error rate.
    trim
        Optional live feed-forward command. When omitted, its contribution is
        treated as zero.
    command
        Writable State receiving the requested controller output. It should
        normally be initialized before transient integration unless ``trim`` is
        supplied.
    minimum
        Optional lower command limit.
    maximum
        Optional upper command limit.

    Public States
    -------------
    error
        Current setpoint-minus-feedback error.
    integral_error
        Integrated error and dynamic iteration variable.
    integral_error_dot
        Time derivative of integrated error.
    proportional
        Current proportional contribution.
    integral
        Current integral contribution.
    derivative
        Current derivative contribution.
    command_bias
        Constant bias established at transient startup for bumpless transfer.
    raw_command
        Command before minimum and maximum limits are applied.
    saturated
        True when the raw command is outside an active command limit.
    """

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
        # Component.setup() records the constructor inputs and converts plain
        # scalar values into States. The public PID-specific States below are
        # created afterward because they are internal controller quantities,
        # not additional constructor inputs.
        self.setup()

        # ------------------------------------------------------------------
        # Public controller diagnostics and dynamics
        # ------------------------------------------------------------------
        #
        # These are ordinary public FullFlow States. Users may track, print,
        # plot, or otherwise inspect them in the same way as any component
        # pressure, temperature, flow rate, or command.
        self.error = State(0.0).add_label(
            f"{self.name}:error"
        )

        # integral_error is the dynamic iteration variable. Its initial value
        # of zero means that no feedback error has accumulated before transient
        # control begins.
        #
        # This is an initial condition for the integral state, not an assumption
        # that the instantaneous control error begins at zero.
        self.integral_error = State(0.0).add_label(
            f"{self.name}:integral_error"
        )

        # The transient solver integrates integral_error using this derivative.
        # evaluate_states() sets it to either the current error or zero when
        # anti-windup temporarily suspends integration.
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

        # raw_command is the requested output before clamping. Comparing it to
        # command makes saturation and tuning behavior easier to diagnose.
        self.raw_command = State(0.0).add_label(
            f"{self.name}:raw_command"
        )

        self.saturated = State(False).add_label(
            f"{self.name}:saturated"
        )

        # command_bias begins unassigned. Its unassigned state indicates that
        # bumpless initialization has not yet occurred.
        #
        # Once assigned at transient startup, it remains fixed unless the user
        # explicitly changes it.
        self.command_bias = State().add_label(
            f"{self.name}:command_bias"
        )

        # ------------------------------------------------------------------
        # Configuration validation
        # ------------------------------------------------------------------

        # A PID must write directly to its command. A derived State is computed
        # from another expression and cannot be assigned by a component.
        if self.command.is_derived:
            raise ValueError(
                f"{self.name}: command must be a writable State, "
                "not a derived State."
            )

        # Validate fixed limits during construction when both are available.
        # The check is repeated during evaluation because minimum and maximum
        # may themselves be live States that change during a simulation.
        if self.minimum.is_assigned and self.maximum.is_assigned:
            if float(self.minimum.value) > float(self.maximum.value):
                raise ValueError(
                    f"{self.name}: minimum must be less than or equal "
                    "to maximum."
                )

        # The PID is inactive during steady-state solving. If the command has
        # not been initialized but trim exists, use trim as the initial command
        # supplied to the steady-state plant.
        #
        # This is not a transient command jump because no previous command value
        # existed. At transient startup, command_bias still ensures bumpless
        # transfer from this initialized command.
        if not self.command.is_assigned and self.trim.is_assigned:
            self.command.value = float(self.trim.value)

    def evaluate_states(self) -> None:
        """Evaluate the PID command and dynamic integral-error derivative.

        This method may be called repeatedly while FullFlow converges a single
        transient timestep. It therefore calculates current trial values but
        does not manually advance accepted controller history.

        Previous accepted values are read through ``State.previous``.
        """

        # ------------------------------------------------------------------
        # Resolve current controller inputs and tuning values
        # ------------------------------------------------------------------

        feedback = float(self.feedback.value)
        setpoint = float(self.setpoint.value)

        proportional_gain = float(self.proportional_gain.value)
        integral_gain = float(self.integral_gain.value)
        derivative_gain = float(self.derivative_gain.value)

        # An omitted trim contributes zero. If trim is a State, its current
        # value is read on every evaluation so it remains a live feed-forward
        # signal.
        trim = (
            float(self.trim.value)
            if self.trim.is_assigned
            else 0.0
        )

        # Omitted command limits remain inactive.
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

        # Limits can be States and may change after construction, so verify
        # their ordering during evaluation as well.
        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"{self.name}: minimum must be less than or equal "
                "to maximum."
            )

        # A writable command must have an initial value before the PID can
        # calculate a bumpless bias. Trim may supply that initial value when
        # the user did not initialize command explicitly.
        if not self.command.is_assigned:
            if self.trim.is_assigned:
                self.command.value = trim
            else:
                raise ValueError(
                    f"{self.name}: PID requires either a trim value or "
                    "an initialized command State."
                )

        # ------------------------------------------------------------------
        # Current PID error and proportional/integral contributions
        # ------------------------------------------------------------------

        error = setpoint - feedback

        proportional = proportional_gain * error

        # integral_error is solved as a dynamic State by FullFlow. Its current
        # value is the solver's present trial value for this timestep.
        integral = (
            integral_gain
            * float(self.integral_error.value)
        )

        # ------------------------------------------------------------------
        # One-time bumpless-transfer initialization
        # ------------------------------------------------------------------

        # At the beginning of transient control, choose a fixed bias such that:
        #
        #     initial command
        #       = trim + bias + proportional + integral
        #
        # The initial derivative is defined as zero because no accepted
        # transient error interval is yet available.
        #
        # This makes startup bumpless whether trim is present or absent.
        if not self.command_bias.is_assigned:
            self.command_bias.value = (
                float(self.command.value)
                - trim
                - proportional
                - integral
            )

        # ------------------------------------------------------------------
        # Timestep and derivative contribution
        # ------------------------------------------------------------------

        time = float(self.network.time.value)

        try:
            # network.time.previous is the time of the previous accepted
            # transient solution, not the time of a previous nonlinear
            # residual evaluation.
            previous_time = float(self.network.time.previous)
            dt = time - previous_time
        except ValueError:
            # No accepted transient history exists at initialization.
            dt = 0.0

        if dt > 0.0:
            try:
                # self.error.previous remains fixed throughout all nonlinear
                # evaluations of the current timestep.
                previous_error = float(self.error.previous)
            except ValueError:
                # No accepted error history exists during the first transient
                # evaluation, so initialize the derivative contribution to zero.
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

        # ------------------------------------------------------------------
        # Unconstrained PID output
        # ------------------------------------------------------------------

        raw_command = (
            trim
            + float(self.command_bias.value)
            + proportional
            + integral
            + derivative
        )

        # ------------------------------------------------------------------
        # Command limiting
        # ------------------------------------------------------------------

        command = raw_command
        saturated = False

        if minimum is not None and command < minimum:
            command = minimum
            saturated = True

        if maximum is not None and command > maximum:
            command = maximum
            saturated = True

        # ------------------------------------------------------------------
        # Conditional-integration anti-windup
        # ------------------------------------------------------------------

        # FullFlow integrates integral_error implicitly. To decide whether the
        # integrator should continue, predict the backward-Euler integral value:
        #
        #     integral_error_new
        #       = integral_error_previous + error_new * dt
        #
        # The prediction uses the previous accepted integral value and the
        # current trial error. It does not permanently modify the integral State.
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
                # During transient initialization there may not yet be an
                # accepted integral value. In that case, retain the current
                # initialized integral state.
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

        # integral_direction is the direction in which continued error
        # integration would move the command:
        #
        #     d(integral contribution)/dt = integral_gain * error
        #
        # Using this quantity instead of only the sign of error makes the
        # anti-windup logic valid for both positive and negative integral gains.
        integral_direction = (
            integral_gain
            * error
        )

        hold_integral = False

        # At the upper limit, suspend integration only when the integral action
        # would increase the command further. Integration remains active when it
        # would reduce the command and help the controller leave saturation.
        if (
            maximum is not None
            and candidate_raw_command > maximum
            and integral_direction > 0.0
        ):
            hold_integral = True

        # At the lower limit, suspend integration only when the integral action
        # would decrease the command further. Integration remains active when it
        # would increase the command back toward the valid range.
        if (
            minimum is not None
            and candidate_raw_command < minimum
            and integral_direction < 0.0
        ):
            hold_integral = True

        # The integral State remains part of the transient solve in either case.
        # Setting its derivative to zero holds its value at the previous accepted
        # timestep; setting it to error performs normal PID error integration.
        self.integral_error_dot.value = (
            0.0
            if hold_integral
            else error
        )

        # ------------------------------------------------------------------
        # Publish controller outputs and diagnostics
        # ------------------------------------------------------------------

        self.error.value = error
        self.proportional.value = proportional
        self.integral.value = integral
        self.derivative.value = derivative

        self.raw_command.value = raw_command
        self.saturated.value = saturated

        # The plant receives the clamped command, while raw_command preserves
        # the unconstrained PID request for diagnostics and tuning.
        self.command.value = command

    @property
    def dynamics(self):
        """Return the PID integral-error dynamic equation.

        FullFlow interprets this pair as:

            d(integral_error)/dt = integral_error_dot

        ``integral_error`` is therefore included as a transient iteration
        variable and integrated together with the physical network.
        """

        return [
            (
                self.integral_error,
                self.integral_error_dot,
            )
        ]