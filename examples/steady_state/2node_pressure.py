"""
Pressurized Gas Volume with Compressible Inlet Flow
===================================================

This example demonstrates a simple pressurization system consisting of:

    High-Pressure Gas Source
              |
              v
    Compressible Flow Tube
              ^
              |
      Churchill Friction Factor
              |
              v
          Gas Volume
              |
              v
     Isentropic Exit Nozzle

The upstream gas source feeds a volume through a compressible flow tube.
The volume pressure is not specified directly and is instead solved from
steady-state mass conservation.

The tube pressure loss is determined using the Darcy-Weisbach equation,
while the friction factor is calculated from the Churchill correlation.

The outlet nozzle expands the flow through an area increase and computes
the resulting supersonic exit state using isentropic compressible-flow
relations.

This example demonstrates:

- Lookup-based thermodynamic properties
- Compressible pipe flow
- Churchill friction-factor calculations
- Solving an unknown node pressure
- Coupling multiple components through shared states
- Supersonic isentropic area expansion

Network Layout
--------------
    GN2 Source
         |
         v
    CompressibleFlowTube
         ^
         |
      Churchill
         |
         v
       Volume
         |
         v
    IsentropicAreaChange

The volume pressure is solved automatically such that:

    mass_flow_in - mass_flow_out = 0

at steady state.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Create network
# -----------------------------------------------------------------------------

VolumeNetwork = Network("Volume Network")


# -----------------------------------------------------------------------------
# High-pressure nitrogen source
#
# This lookup provides thermodynamic properties for the supply bottle.
#
# Outputs such as:
#
#     SourceFluid.pressure
#     SourceFluid.density
#     SourceFluid.dynamic_viscosity
#
# can be connected directly into downstream components.
# -----------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source Fluid",
    VolumeNetwork,
    IdealGas,
    fluid="gn2",
    pressure=4.2e7,      # Pa
    temperature=300,     # K
)


# -----------------------------------------------------------------------------
# Volume gas state
#
# This lookup represents the thermodynamic state inside the gas volume.
#
# The pressure is connected to the volume pressure state and will be updated
# automatically as the solver iterates.
# -----------------------------------------------------------------------------

VolumeFluid = Lookup(
    "Volume Fluid",
    VolumeNetwork,
    IdealGas,
    fluid=SourceFluid.fluid,
    pressure=3e7,        # Initial pressure guess [Pa]
    temperature=300,     # K
)


# -----------------------------------------------------------------------------
# Compressible inlet tube
#
# Solves compressible flow through a constant-area tube with friction.
#
# Unknown:
#     mass_flow
#
# The downstream pressure is coupled directly to the volume pressure, making
# the tube and volume part of the same nonlinear system.
# -----------------------------------------------------------------------------

Tube = FlowTube(
    "Pressurant Tube",
    VolumeNetwork,
    mass_flow=30,  # Initial guess [kg/s]
    upstream_static_pressure=SourceFluid.pressure,
    upstream_density=SourceFluid.density,
    downstream_static_pressure=VolumeFluid.pressure,
    downstream_density=VolumeFluid.density,
    length=10,                 # m
    hydraulic_diameter=1 / 39.3701,  # 1 inch [m]
    friction_factor=0.002,     # Initial guess
)


# -----------------------------------------------------------------------------
# Churchill friction-factor correlation
#
# Computes the Darcy friction factor based on Reynolds number and roughness.
#
# This component is coupled to the tube through:
#
#     Tube.mass_flow
#     Tube.friction_factor
# -----------------------------------------------------------------------------

TubeFriction = Churchill(
    "Tube Churchill Friction Factor",
    VolumeNetwork,
    mass_flow=Tube.mass_flow,
    friction_factor=Tube.friction_factor,
    hydraulic_diameter=Tube.hydraulic_diameter,
    cross_sectional_area=(np.pi / 4) * (1 / 39.3701) ** 2,
    dynamic_viscosity=SourceFluid.dynamic_viscosity,
    roughness=1e-7,  # m
)


# -----------------------------------------------------------------------------
# Gas volume
#
# The volume pressure is the primary unknown in this example.
#
# The steady-state solver adjusts pressure until:
#
#     mass_flow_in = mass_flow_out
#
# Since no energy terms are supplied, Volume automatically operates in
# mass-conservation mode.
# -----------------------------------------------------------------------------

GasVolume = Volume(
    "Gas Volume",
    VolumeNetwork,
    pressure=VolumeFluid.pressure,
    volume=1.0,                 # m^3
    mass_flow_in=Tube.mass_flow,
    mass_flow_out=10
)










# -----------------------------------------------------------------------------
# Solve network
# -----------------------------------------------------------------------------

SteadyState(VolumeNetwork).solve(
    verbose=True,
    print_solution=True,
)