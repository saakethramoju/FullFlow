"""
Transient Point Mass With Vector and Matrix Expressions
======================================================

This example combines thrust and gravity to calculate force and moment in
the body frame, then rotates the force into the inertial frame to integrate
a point mass's position and velocity.

StateArray groups scalar States into vectors. Matrix products, cross products,
and norms remain live expressions as their input States change. The rotation
matrix is one array-valued State and stays fixed throughout this solve.
"""

import numpy as np
from fullflow import *


filename = "3vector_math"
dt = 0.1
t_final = 1.0


class PointMass(Component):
    def __init__(self, name, network, mass, force, position, velocity):
        self.setup()

    @property
    def dynamics(self):
        return [
            (self.position, self.velocity),
            (self.velocity, self.force / self.mass),
        ]


# ---------------------------------------------------------------------------
# Network and applied loads
# ---------------------------------------------------------------------------

Vehicle = Network("Vector Math")

mass = State(10.0)
thrust = State(120.0)
rotation = State(np.eye(3))  # Body-to-inertial rotation; frames initially align.

thrust_body = StateArray([State(0.0), State(0.0), thrust])
gravity_inertial = StateArray([State(0.0), State(0.0), -9.81 * mass])

thrust_position = StateArray([State(0.2), State(0.0), State(-1.0)])
center_of_mass = StateArray([State(0.0), State(0.1), State(0.0)])

force_body = StateArray.full((3,), 0.0)
moment_body = StateArray.full((3,), 0.0)

# Bind each vector to an expression that updates with its inputs.
force_body <<= thrust_body + rotation.T @ gravity_inertial
# Calculate the moment for tracking without integrating rotational motion.
moment_body <<= (
    thrust_position.cross(thrust_body)
    + center_of_mass.cross(rotation.T @ gravity_inertial)
)

force_inertial = rotation @ force_body
force_magnitude = force_body.norm()


# ---------------------------------------------------------------------------
# Components and initial conditions
# ---------------------------------------------------------------------------

position = StateArray.full((3,), 0.0)
velocity = StateArray.full((3,), 0.0)

Motion = PointMass(
    "Point Mass",
    Vehicle,
    mass=mass,
    force=force_inertial,
    position=position,
    velocity=velocity,
)


# ---------------------------------------------------------------------------
# Outputs and transient solve
# ---------------------------------------------------------------------------

Vehicle.track("Force Body [N]", force_body)
Vehicle.track("Moment Body [N m]", moment_body)
Vehicle.track("Force Magnitude [N]", force_magnitude)
Vehicle.track("Position [m]", position)
Vehicle.track("Velocity [m/s]", velocity)

Transient(Vehicle).solve(
    dt=dt,
    t_final=t_final,
    filename=filename,
    history="summary",
)

print("Body force [N]:", force_body.value)
print("Body moment [N m]:", moment_body.value)
print("Final position [m]:", position.value)
print("Final velocity [m/s]:", velocity.value)
