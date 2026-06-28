"""
Simple reciprocating piston-cylinder example.

Physical layout
---------------

    Moving piston
        |
        v

    +-----------------------------+
    |                             |
    |        Nitrogen gas          |  closed system
    |                             |
    +-----------------------------+

The piston motion prescribes the cylinder volume:

    L(t) = 4 in + 3 in * cos(omega * t)
    V(t) = A_piston * L(t)

There are no inlet or outlet mass flows.

FullFlow solves the gas pressure and temperature from:

    mass = density(P, T) * volume

    total_internal_energy = mass * internal_energy(P, T)

    d(total_internal_energy)/dt = -P * dV/dt

For compression, dV/dt is negative, so -P*dV/dt is positive and work is added
to the gas.
"""

import math

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Unit conversions
# -----------------------------------------------------------------------------

IN_TO_M = 0.0254
PSIA_TO_PA = 6894.757293
F_TO_K_OFFSET = 459.67
R_TO_K = 5.0 / 9.0


# -----------------------------------------------------------------------------
# GFSSP Example 9 style inputs
# -----------------------------------------------------------------------------

diameter = 3.0 * IN_TO_M
area = math.pi / 4.0 * diameter**2

mean_length = 4.0 * IN_TO_M
stroke_amplitude = 3.0 * IN_TO_M

rpm = 1200.0
omega = 2.0 * math.pi * rpm / 60.0

initial_pressure = 14.7 * PSIA_TO_PA
initial_temperature = (75.0 + F_TO_K_OFFSET) * R_TO_K

initial_length = mean_length + stroke_amplitude
initial_volume = area * initial_length

gamma = 1.4


def cylinder_volume_command(t):
    """Prescribed piston-cylinder volume."""
    length = mean_length + stroke_amplitude * math.cos(omega * t)
    return area * length


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

PistonCylinder = Network("Reciprocating Piston Cylinder")


# The volume is an imposed geometric input.  The solver does not solve for this;
# the Sequence updates it from the piston kinematics.
CylinderVolume = State(initial_volume)

CylinderVolumeSequence = Sequence(
    "Cylinder Volume Sequence",
    PistonCylinder,
    target=CylinderVolume,
    function=cylinder_volume_command,
)


# Pressure and temperature are the dynamic solve states.
CylinderPressure = State(initial_pressure)
CylinderTemperature = State(initial_temperature)


# Nitrogen properties at the current cylinder pressure and temperature.
CylinderGas = Lookup(
    "Cylinder Gas",
    PistonCylinder,
    Fluid,
    "nitrogen",
    pressure=CylinderPressure,
    temperature=CylinderTemperature,
)


# Closed moving-volume gas node.
Cylinder = Volume(
    "Cylinder",
    PistonCylinder,
    pressure=CylinderPressure,
    volume=CylinderVolume,
    density=CylinderGas.density,
    temperature=CylinderTemperature,
    enthalpy=CylinderGas.enthalpy,
    internal_energy=CylinderGas.internal_energy,
    energy_variable="T",
    work_pressure=CylinderPressure,
)


# Simple isentropic reference using constant gamma.
IsentropicPressure = initial_pressure * (initial_volume / CylinderVolume)**gamma
IsentropicTemperature = initial_temperature * (initial_volume / CylinderVolume)**(gamma - 1.0)


# -----------------------------------------------------------------------------
# Tracks
# -----------------------------------------------------------------------------

PistonCylinder.track("Cylinder Volume [m3]", CylinderVolume)
PistonCylinder.track("Cylinder Pressure [Pa]", CylinderPressure)
PistonCylinder.track("Cylinder Temperature [K]", CylinderTemperature)
PistonCylinder.track("Isentropic Pressure [Pa]", IsentropicPressure)
PistonCylinder.track("Isentropic Temperature [K]", IsentropicTemperature)
PistonCylinder.track("Boundary Work Rate [W]", Cylinder.boundary_work_rate)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

# Do not run SteadyState first for this simple closed cylinder.
# At t=0, pressure and temperature are the initial condition, not the result of
# a steady-flow balance.
Transient(PistonCylinder).solve(
    dt=1.0e-4,
    t_final=0.10,
    save_dt=1.0e-4,
    filename="reciprocating_piston_cylinder",
    verbose=True,
)