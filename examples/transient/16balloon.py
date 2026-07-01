"""
Simple balloon deflation example.

Physical layout
---------------

        Elastic balloon wall
    +-------------------------+
    |                         |
    |       Air / gas          |
    |                         |
    +------------+------------+
                 |
                 v
          Neck / outlet line
          with friction and inertia
                 |
                 v
          Ambient pressure

The balloon volume is not prescribed directly.  Instead, the balloon volume is
a function of balloon pressure:

    V_balloon = V_relaxed + C_balloon * (P_balloon - P_ambient)

The gas node stores mass and energy.  As gas leaves through the neck, pressure
falls, the balloon shrinks, and boundary work is included through the changing
volume.

The neck is modeled with DarcyWeisbach, so the mass flow has line inertia.  To
avoid an artificial startup spike, the initial neck mass flow is estimated from
the initial Darcy pressure-drop relation.
"""

import math

from fullflow import *


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ambient_pressure = 101325.0       # Pa
initial_pressure = 130000.0       # Pa
initial_temperature = 300.0       # K

gas_constant = 287.05             # J/kg-K, air
specific_heat_ratio = 1.4
specific_heat_cv = gas_constant / (specific_heat_ratio - 1.0)
specific_heat_cp = specific_heat_ratio * specific_heat_cv

relaxed_volume = 0.002            # m^3
initial_volume = 0.010            # m^3

balloon_compliance = (initial_volume - relaxed_volume) / (initial_pressure - ambient_pressure)

neck_diameter = 0.006             # m
neck_area = math.pi / 4.0 * neck_diameter**2
neck_length = 0.05                # m
neck_friction_factor = 0.04


# -----------------------------------------------------------------------------
# Initial neck mass-flow estimate
# -----------------------------------------------------------------------------

initial_density = initial_pressure / (gas_constant * initial_temperature)
initial_pressure_drop = initial_pressure - ambient_pressure

if initial_pressure_drop == 0.0:
    initial_neck_mass_flow = 0.0
else:
    initial_neck_mass_flow = math.copysign(
        neck_area
        * math.sqrt(
            2.0
            * initial_density
            * neck_diameter
            * abs(initial_pressure_drop)
            / (neck_friction_factor * neck_length)
        ),
        initial_pressure_drop,
    )


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

BalloonNetwork = Network("Balloon Deflation")


# Boundary pressure.
AmbientPressure = State(ambient_pressure)


# Balloon gas states.
BalloonPressure = State(initial_pressure)
BalloonTemperature = State(initial_temperature)


# Ideal-gas properties using State math.
BalloonDensity = BalloonPressure / (gas_constant * BalloonTemperature)
BalloonInternalEnergy = specific_heat_cv * BalloonTemperature
BalloonEnthalpy = specific_heat_cp * BalloonTemperature


# Elastic balloon pressure-volume relation.
BalloonVolume = relaxed_volume + balloon_compliance * (BalloonPressure - AmbientPressure)


# Neck mass flow.
NeckMassFlow = State(initial_neck_mass_flow)


Neck = DarcyWeisbach(
    "Balloon Neck",
    BalloonNetwork,
    mass_flow=NeckMassFlow,
    upstream_pressure=BalloonPressure,
    downstream_pressure=AmbientPressure,
    length=neck_length,
    hydraulic_diameter=neck_diameter,
    density=BalloonDensity,
    cross_sectional_area=neck_area,
    friction_factor=neck_friction_factor,
)


Balloon = Volume(
    "Balloon",
    BalloonNetwork,
    pressure=BalloonPressure,
    volume=BalloonVolume,
    density=BalloonDensity,
    temperature=BalloonTemperature,
    enthalpy=BalloonEnthalpy,
    internal_energy=BalloonInternalEnergy,
    mass_flow_out=NeckMassFlow,
    energy_variable="temperature",
    work_pressure=BalloonPressure,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

BalloonNetwork.track("Balloon Pressure [Pa]", BalloonPressure)
BalloonNetwork.track("Balloon Temperature [K]", BalloonTemperature)
BalloonNetwork.track("Balloon Volume [m3]", BalloonVolume)
BalloonNetwork.track("Neck Mass Flow [kg/s]", NeckMassFlow)
BalloonNetwork.track("Balloon Density [kg/m3]", BalloonDensity)
BalloonNetwork.track("Balloon Boundary Work Rate [W]", Balloon.boundary_work_rate)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

# Do not steady-state initialize this case.  The inflated balloon is the initial
# condition.  A steady-state solve would try to drive the open balloon toward
# ambient pressure and zero flow.
BalloonSolver = Transient(BalloonNetwork)

BalloonSolver.solve(
    dt=0.001,
    t_final=1.0,
    save_dt=0.001,
    filename="16balloon",
    verbose=True,
)