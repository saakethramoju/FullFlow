"""
Sampled LQR stabilization of a rocket using thrust-vector control.

Physical layout
---------------

    Pitch setpoint (0 deg)       Rocket pitch, pitch rate
              |                           |
              +----------+----------------+
                         |
                         v
                  Discrete LQR (20 Hz)
                         |
                         v
                  Gimbal actuator
                  +/- 6 deg, slew limit
                         |
                         v
                Thrust moment, F * L * sin(delta)
                         |
                         v
                 Rocket pitch dynamics

Model notes
-----------

The discrete feedback gain comes from the sampled, zero-order-held linearization
of the pitch dynamics. The FullFlow plant uses the nonlinear sine of the actual
gimbal angle, not the linearized control input. This models small pitch motions
and a fixed thrust/inertia operating point, not a full six-degree-of-freedom flight.
"""

import math

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import fullplot as fplt
from fullflow import *


# -----------------------------------------------------------------------------
# Run options
# -----------------------------------------------------------------------------

dt = 0.05
t_final = 6.0
filename = '29thrust_vectoring_lqr'
sample_rate = 20.0


# -----------------------------------------------------------------------------
# Rocket parameters
# -----------------------------------------------------------------------------

thrust = 18000.0
lever_arm = 1.8
pitch_inertia = 1200.0
pitch_damping = 0.5
max_gimbal = math.radians(6.0)
max_gimbal_rate = math.radians(25.0)
moment_gain = thrust * lever_arm / pitch_inertia


# -----------------------------------------------------------------------------
# Discrete LQR design
# -----------------------------------------------------------------------------

A = np.array([[0.0, 1.0], [0.0, -pitch_damping]])
B = np.array([[0.0], [moment_gain]])
Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), 1.0 / sample_rate)
Q = np.diag([40.0, 1.0])
R = np.array([[10.0]])
P = solve_discrete_are(Ad, Bd, Q, R)
K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad).ravel()


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Rocket = Network("Rocket Pitch Stabilization")
pitch = State(math.radians(6.0))
pitch_rate = State(math.radians(-1.0))
gimbal_command = State(0.0)
gimbal_position = State(0.0)


class PitchDynamics(Component):
    def __init__(self, name, network, pitch, pitch_rate, gimbal, acceleration=None):
        self.setup()

    def evaluate_states(self):
        self.acceleration.value = (
            moment_gain * math.sin(self.gimbal.value)
            - pitch_damping * self.pitch_rate.value
        )

    @property
    def dynamics(self):
        return [(self.pitch, self.pitch_rate), (self.pitch_rate, self.acceleration)]


def lqr(t, angle, angular_rate):
    return float(np.clip(-K @ np.array([angle, angular_rate]), -max_gimbal, max_gimbal))


Controller = Sequence(
    "Pitch LQR",
    Rocket,
    target=gimbal_command,
    function=lqr,
    inputs=[pitch, pitch_rate],
    sample_rate=sample_rate,
)

Gimbal = Actuator(
    "Nozzle Gimbal",
    Rocket,
    command=gimbal_command,
    position=gimbal_position,
    minimum=-max_gimbal,
    maximum=max_gimbal,
    rate_limit=max_gimbal_rate,
)

Vehicle = PitchDynamics(
    "Pitch Dynamics",
    Rocket,
    pitch=pitch,
    pitch_rate=pitch_rate,
    gimbal=gimbal_position,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

Rocket.track("Pitch [deg]", pitch * 180.0 / math.pi)
Rocket.track("Pitch Rate [deg/s]", pitch_rate * 180.0 / math.pi)
Rocket.track("Gimbal Command [deg]", gimbal_command * 180.0 / math.pi)
Rocket.track("Gimbal Position [deg]", gimbal_position * 180.0 / math.pi)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

Transient(Rocket).solve(
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
    "Rocket_Pitch_Stabilization/transient/tracks"
)
results.plot(
    y=[
        results.trace(y="Pitch [deg]", x="time"),
        results.trace(y="Pitch Rate [deg/s]", x="time"),
    ],
    y2=[
        results.trace(y="Gimbal Command [deg]", x="time"),
        results.trace(y="Gimbal Position [deg]", x="time"),
    ],
    xlabel="Time [s]",
    title="Rocket Pitch LQR Stabilization",
)
fplt.show()
