"""
Pipe Flow with Darcy-Weisbach and Colebrook Friction Factor
===========================================================

This example demonstrates a simple incompressible pipe flow calculation using
FullFlow and ThermoProp.

Network Layout
--------------
    Water Source
          |
          v
    Darcy-Weisbach Pipe
          ^
          |
      Colebrook

The Darcy-Weisbach component solves for the pipe mass flow rate using the
pressure drop, fluid density, geometry, and friction factor.

The Colebrook component simultaneously solves for the friction factor based on
the current Reynolds number and pipe roughness.

Together, the two components form a coupled nonlinear system:

    Darcy-Weisbach:
        ΔP = f (L/D) (ρV²/2)

    Colebrook:
        1/√f = -2 log10(ε/(3.7D) + 2.51/(Re√f))

The steady-state solver iterates on both mass flow rate and friction factor
until both equations are satisfied simultaneously.

Demonstrates
------------
- ThermoProp property lookups using Lookup
- Coupled iteration variables
- Darcy-Weisbach pressure-loss calculations
- Colebrook friction-factor calculations
- Automatic steady-state network solving
"""

import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Create network
# -----------------------------------------------------------------------------

PipeNetwork = Network("Pipe Network")


# -----------------------------------------------------------------------------
# Fluid property lookup
#
# ThermoProp provides density, viscosity, and other properties that are used
# by downstream components. Outputs such as:
#
#     SourceFluid.density
#     SourceFluid.dynamic_viscosity
#
# behave like FullFlow states and can be connected directly into components.
# -----------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source Fluid",
    PipeNetwork,
    Fluid,
    "Water",
    pressure=3e5,      # Pa
    temperature=300,   # K
)


# -----------------------------------------------------------------------------
# Pipe geometry
#
# Diameter = 0.5 mm
#
# FullFlow uses SI units internally.
# -----------------------------------------------------------------------------

pipe_diameter = 0.5 / 1550  # m
pipe_area = (np.pi / 4) * pipe_diameter**2


# -----------------------------------------------------------------------------
# Darcy-Weisbach pipe
#
# Solves the pressure-loss equation for mass flow rate.
#
# Unknown:
#     mass_flow
#
# Uses:
#     upstream pressure
#     downstream pressure
#     density
#     friction factor
#     geometry
# -----------------------------------------------------------------------------

Pipe = DarcyWeisbach(
    "Pipe",
    PipeNetwork,
    mass_flow=3,  # Initial guess [kg/s]
    upstream_pressure=SourceFluid.pressure,
    downstream_pressure=101325,  # Atmospheric pressure [Pa]
    length=5,                    # Pipe length [m]
    cross_sectional_area=pipe_area,
    hydraulic_diameter=pipe_diameter,
    density=SourceFluid.density,
    friction_factor=0.002,       # Initial guess
)


# -----------------------------------------------------------------------------
# Colebrook friction-factor solver
#
# Solves for the Darcy friction factor based on Reynolds number and roughness.
#
# Unknown:
#     friction_factor
#
# Coupled to the Darcy-Weisbach pipe through:
#
#     Pipe.mass_flow
#     Pipe.friction_factor
# -----------------------------------------------------------------------------

PipeFriction = Colebrook(
    "Pipe Colebrook Friction Factor",
    PipeNetwork,
    mass_flow=Pipe.mass_flow,
    friction_factor=Pipe.friction_factor,
    hydraulic_diameter=Pipe.hydraulic_diameter,
    dynamic_viscosity=SourceFluid.dynamic_viscosity,
    cross_sectional_area=Pipe.cross_sectional_area,
    roughness=1e-6,  # m
)


# -----------------------------------------------------------------------------
# Solve network
# -----------------------------------------------------------------------------

SteadyState(PipeNetwork).solve(verbose=True)