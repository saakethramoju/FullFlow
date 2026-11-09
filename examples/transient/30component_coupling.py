"""
Coupled tank and vehicle dynamics using direct Component references.

Physical layout
---------------

    Propellant Tank  ------ mass, outlet flow ------>  Vehicle
         ^                                         |
         |                                         |
         +------------ vehicle acceleration -------+

The tank supplies propellant mass and thrust-producing flow. Vehicle acceleration
changes the tank's effective hydrostatic head, so the two Components depend on
one another within the same implicit transient solve.

This is a simplified vertical-flight example with a constant ullage pressure,
constant fluid density implicit in the head relation, and no drag or sloshing.
"""

import math

import fullplot as fplt
from fullflow import *


# -----------------------------------------------------------------------------
# Run options
# -----------------------------------------------------------------------------

dt = 0.02
t_final = 1.0
filename = '30component_coupling'


# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------

gravity = 9.80665
ambient_pressure = 101325.0


# -----------------------------------------------------------------------------
# Custom components
# -----------------------------------------------------------------------------

class PropellantTank(Component):
    def __init__(self, name, network, mass, ullage_pressure, tank_area, flow_coefficient):
        self.outlet_pressure = State(0.0)
        self.mass_flow = State(0.0)
        self.mass_dot = 0.0
        self.setup()

    def evaluate_states(self):
        effective_g = gravity + self.vehicle.acceleration.value
        self.outlet_pressure.value = (
            self.ullage_pressure.value
            + self.mass.value / self.tank_area.value * effective_g
        )
        self.mass_flow.value = self.flow_coefficient.value * math.sqrt(
            max(self.outlet_pressure.value - ambient_pressure, 0.0)
        )
        self.mass_dot = -self.mass_flow.value

    @property
    def dynamics(self):
        return [(self.mass, self.mass_dot)]


class Vehicle(Component):
    def __init__(self, name, network, fuel_tank, dry_mass, exhaust_velocity, altitude, velocity):
        self.acceleration = State(0.0)
        self.total_mass = State(0.0)
        self.altitude_dot = 0.0
        self.velocity_dot = 0.0
        self.setup()

    def evaluate_states(self):
        self.total_mass.value = self.dry_mass.value + self.fuel_tank.mass.value
        thrust = self.fuel_tank.mass_flow.value * self.exhaust_velocity.value
        self.acceleration.value = thrust / self.total_mass.value - gravity
        self.altitude_dot = self.velocity.value
        self.velocity_dot = self.acceleration.value

    @property
    def dynamics(self):
        return [
            (self.altitude, self.altitude_dot),
            (self.velocity, self.velocity_dot),
        ]


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Flight = Network("Coupled Vehicle and Tank")

Tank = PropellantTank(
    "Fuel Tank",
    Flight,
    mass=State(10.0),
    ullage_pressure=200000.0,
    tank_area=0.1,
    flow_coefficient=0.002,
)

Rocket = Vehicle(
    "Rocket",
    Flight,
    fuel_tank=Tank,
    dry_mass=100.0,
    exhaust_velocity=2500.0,
    altitude=State(0.0),
    velocity=State(0.0),
)

Tank.vehicle = Rocket


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

Flight.track("Tank Mass [kg]", Tank.mass)
Flight.track("Tank Outlet Pressure [Pa]", Tank.outlet_pressure)
Flight.track("Vehicle Mass [kg]", Rocket.total_mass)
Flight.track("Altitude [m]", Rocket.altitude)
Flight.track("Acceleration [m/s2]", Rocket.acceleration)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

Transient(Flight).solve(
    dt=dt,
    t_final=t_final,
    verbose=True,
    statistics=True,
    filename=filename,
)