"""
Spacecraft reaction-control thrusters with localized bang-bang switching.

Physical layout
---------------

                         Cold GN2 bottle
                         80 bar, 10 L
                              |
                    +---------+---------+
                    |                   |
              Pitch jet pair        Yaw jet pair
              (+/- torque)          (+/- torque)
                    |                   |
                    +---------+---------+
                              |
                       Spacecraft body
                   pitch / yaw dynamics
                              |
                    +---------+---------+
                    |                   |
                Pitch relay          Yaw relay
                    ^                   ^
                    |                   |
                 Attitude and rate feedback
                    ^                   ^
                    |                   |
                 Mission pointing schedules

Each axis uses a three-position relay: fire a positive jet, coast, or fire a
negative jet. Hysteresis prevents rapid switching near the deadband. Both
controllers use ``event=True`` to locate switch times within a timestep.

The jets share a finite gas supply, so firing either axis reduces the thrust
available to the other. The bottle is modeled as an ideal, isothermal gas and
the jets as ideal choked restrictions; valve-opening dynamics are neglected.
"""

import math

import fullplot as fplt
from fullflow import *


# -----------------------------------------------------------------------------
# Run options
# -----------------------------------------------------------------------------

dt = 0.25
t_final = 18.0
filename = '27reaction_control'


# -----------------------------------------------------------------------------
# Spacecraft and RCS parameters
# -----------------------------------------------------------------------------

gas_constant = 296.8
bottle_temperature = 295.0
bottle_volume = 0.010
initial_pressure = 8.0e6
heat_capacity_ratio = 1.4
discharge_coefficient = 0.85
jet_area = 1.8e-6
specific_impulse = 68.0
lever_arm = 0.75
pitch_inertia = 130.0
yaw_inertia = 145.0
pitch_damping = 0.8
yaw_damping = 1.0
lead_time = 0.8
switch_on = math.radians(0.6)
switch_off = math.radians(0.12)
jet_flow_factor = (
    discharge_coefficient * jet_area
    * math.sqrt(heat_capacity_ratio / (gas_constant * bottle_temperature))
    * (2.0 / (heat_capacity_ratio + 1.0)) ** (
        (heat_capacity_ratio + 1.0) / (2.0 * (heat_capacity_ratio - 1.0))
    )
)


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Spacecraft = Network("Spacecraft Reaction Control")
bottle_pressure = State(initial_pressure)
pitch = State(math.radians(8.0))
pitch_rate = State(math.radians(-0.4))
yaw = State(math.radians(-6.0))
yaw_rate = State(math.radians(0.3))
pitch_target = State(0.0)
yaw_target = State(0.0)
pitch_jet = State(0.0)
yaw_jet = State(0.0)


# -----------------------------------------------------------------------------
# Guidance and control
# -----------------------------------------------------------------------------

Sequence(
    "Pitch Pointing Program",
    Spacecraft,
    target=pitch_target,
    times=[0.0, 6.0, 6.3, 12.0, 12.3, t_final],
    values=[0.0, 0.0, math.radians(3.0), math.radians(3.0), 0.0, 0.0],
)

Sequence(
    "Yaw Pointing Program",
    Spacecraft,
    target=yaw_target,
    times=[0.0, 6.0, 6.3, 12.0, 12.3, t_final],
    values=[0.0, 0.0, math.radians(-2.0), math.radians(-2.0), 0.0, 0.0],
)


def thruster_control(t, angle, rate, target, previous_jet):
    pointing_error = angle - target + lead_time * rate
    if pointing_error > switch_on:
        return -1.0
    if pointing_error < -switch_on:
        return 1.0
    if abs(pointing_error) < switch_off:
        return 0.0
    return previous_jet


Sequence(
    "Pitch RCS",
    Spacecraft,
    target=pitch_jet,
    function=thruster_control,
    inputs=[pitch, pitch_rate, pitch_target, pitch_jet],
    event=True,
)

Sequence(
    "Yaw RCS",
    Spacecraft,
    target=yaw_jet,
    function=thruster_control,
    inputs=[yaw, yaw_rate, yaw_target, yaw_jet],
    event=True,
)


# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

class ReactionControlSystem(Component):
    def __init__(
        self,
        name,
        network,
        pressure,
        pitch,
        pitch_rate,
        yaw,
        yaw_rate,
        pitch_command,
        yaw_command,
        pressure_rate=None,
        pitch_acceleration=None,
        yaw_acceleration=None,
        mass_flow=None,
        jet_thrust=None,
    ):
        self.setup()

    def evaluate_states(self):
        p = max(self.pressure.value, 0.0)
        jet_flow = jet_flow_factor * p
        active_jets = abs(self.pitch_command.value) + abs(self.yaw_command.value)
        self.mass_flow.value = active_jets * jet_flow
        self.jet_thrust.value = jet_flow * specific_impulse * 9.80665
        self.pressure_rate.value = (
            -gas_constant * bottle_temperature / bottle_volume * self.mass_flow.value
        )
        self.pitch_acceleration.value = (
            lever_arm * self.jet_thrust.value * self.pitch_command.value
            - pitch_damping * self.pitch_rate.value
        ) / pitch_inertia
        self.yaw_acceleration.value = (
            lever_arm * self.jet_thrust.value * self.yaw_command.value
            - yaw_damping * self.yaw_rate.value
        ) / yaw_inertia

    @property
    def dynamics(self):
        return [
            (self.pressure, self.pressure_rate),
            (self.pitch, self.pitch_rate),
            (self.pitch_rate, self.pitch_acceleration),
            (self.yaw, self.yaw_rate),
            (self.yaw_rate, self.yaw_acceleration),
        ]


RCS = ReactionControlSystem(
    "Cold Gas RCS",
    Spacecraft,
    pressure=bottle_pressure,
    pitch=pitch,
    pitch_rate=pitch_rate,
    yaw=yaw,
    yaw_rate=yaw_rate,
    pitch_command=pitch_jet,
    yaw_command=yaw_jet,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

deg = 180.0 / math.pi
Spacecraft.track("Pitch [deg]", pitch * deg)
Spacecraft.track("Pitch Target [deg]", pitch_target * deg)
Spacecraft.track("Yaw [deg]", yaw * deg)
Spacecraft.track("Yaw Target [deg]", yaw_target * deg)
Spacecraft.track("Pitch Rate [deg/s]", pitch_rate * deg)
Spacecraft.track("Yaw Rate [deg/s]", yaw_rate * deg)
Spacecraft.track("Pitch Jet Command", pitch_jet)
Spacecraft.track("Yaw Jet Command", yaw_jet)
Spacecraft.track("Bottle Pressure [bar]", bottle_pressure / 1.0e5)
Spacecraft.track("Total GN2 Flow [kg/s]", RCS.mass_flow)
Spacecraft.track("Jet Thrust [N]", RCS.jet_thrust)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

Transient(Spacecraft).solve(
    dt=dt,
    t_final=t_final,
    verbose=True,
    statistics=True,
    filename=filename,
)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

results = fplt.open(filename).at(
    "Spacecraft_Reaction_Control/transient/runs/base/tracks"
)

results.plot(
    y=[
        results.trace(y="Pitch [deg]", x="time"),
        results.trace(y="Pitch Target [deg]", x="time"),
        results.trace(y="Yaw [deg]", x="time"),
        results.trace(y="Yaw Target [deg]", x="time"),
    ],
    xlabel="Time [s]",
    title="Two-Axis RCS",
)

results.plot(
    y=results.trace(y="Bottle Pressure [bar]", x="time"),
    y2=results.trace(y="Total GN2 Flow [kg/s]", x="time"),
    xlabel="Time [s]",
    title="Shared Cold-Gas Supply Blowdown",
)

fplt.show()
