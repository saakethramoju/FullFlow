"""
Transient Darcy-Weisbach Pipe With Colebrook Friction
====================================================

This example models transient liquid flow through a single pipe connected to a
constant-pressure source and a fixed downstream pressure boundary.

The pipe mass flow is treated as a transient state. During the transient solve,
the Darcy-Weisbach component integrates the pipe momentum equation while the
Colebrook component continuously updates the pipe friction factor from the
current Reynolds number.

Physical Layout
---------------

    Constant-Pressure Source                    Fixed Outlet Boundary
    P = 300 kPa, T = 300 K                       P = 101325 Pa
              │                                      │
              ▼                                      ▼
        ┌──────────┐       Darcy-Weisbach       ┌──────────┐
        │  Source  │ ───── Pipe With Inertia ──▶│  Outlet  │
        └──────────┘                            └──────────┘
                                │
                                ▼
                       Colebrook Friction
                    f = f(Re, D, μ, A, mdot)

Model Notes
-----------
- Source fluid properties are calculated with ThermoProp.
- The pipe mass flow starts from zero and rises toward its steady-state value.
- The Colebrook component updates the Darcy friction factor every timestep.
- The transient solver uses fixed-step implicit backward Euler.
- Output is written to ``simple_pipe.h5``.
"""

import math

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
# A Network is the container that owns all components, states, balances, and
# tracked outputs for this model.
PipeNetwork = Network("Pipe Network")


# ---------------------------------------------------------------------------
# Source Fluid
# ---------------------------------------------------------------------------
# The source is modeled as a fixed thermodynamic state.  ThermoProp calculates
# water properties from pressure and temperature, including density and dynamic
# viscosity.  These properties are then passed into the pipe and friction model.
Source = Lookup(
    "Source",
    PipeNetwork,
    Fluid,
    "water",
    pressure=3e5,       # Pa
    temperature=300,    # K
)


# ---------------------------------------------------------------------------
# Pipe Geometry
# ---------------------------------------------------------------------------
# The example uses a 0.5 inch diameter pipe with a 5 meter length.
pipe_diameter = 0.5 / 39.37
pipe_area = (math.pi / 4) * pipe_diameter**2
pipe_length = 5


# ---------------------------------------------------------------------------
# Transient Pipe Momentum Component
# ---------------------------------------------------------------------------
# DarcyWeisbach computes the pressure-loss / momentum balance for pipe flow.
# Since mass_flow is initialized as a number, FullFlow automatically creates it
# as a State owned by this component.
#
# During a transient solve, Pipe.mass_flow is the integrated transient variable.
# The pressure difference accelerates the liquid until the Darcy-Weisbach loss
# balances the available pressure drop.
Pipe = DarcyWeisbach(
    "Pipe",
    network=PipeNetwork,
    mass_flow=0,                         # kg/s, initial transient guess/state
    upstream_pressure=Source.pressure,   # Pa
    downstream_pressure=101325,          # Pa
    length=pipe_length,                  # m
    hydraulic_diameter=pipe_diameter,    # m
    density=Source.density,              # kg/m^3
    cross_sectional_area=pipe_area,      # m^2
    friction_factor=2e-5,                # initial guess; updated by Colebrook
)


# ---------------------------------------------------------------------------
# Colebrook Friction Component
# ---------------------------------------------------------------------------
# Colebrook updates Pipe.friction_factor using the current pipe mass flow.
# Because friction_factor=Pipe.friction_factor is shared, this component writes
# directly to the same State used by the DarcyWeisbach pipe.
#
# As Pipe.mass_flow changes during the transient, Reynolds number changes, so
# the friction factor is recalculated at each solver evaluation/timestep.
PipeFriction = Colebrook(
    "Pipe Friction",
    PipeNetwork,
    mass_flow=Pipe.mass_flow,
    friction_factor=Pipe.friction_factor,
    hydraulic_diameter=Pipe.hydraulic_diameter,
    dynamic_viscosity=Source.dynamic_viscosity,
    cross_sectional_area=Pipe.cross_sectional_area,
)


# ---------------------------------------------------------------------------
# Optional Output Tracks
# ---------------------------------------------------------------------------
# Tracks give convenient named outputs inside the HDF5 file.  The solver still
# saves the full model state at saved timesteps; these tracks are just useful
# for quick plotting and post-processing.
PipeNetwork.track("Pipe Mass Flow [kg/s]", Pipe.mass_flow)
PipeNetwork.track("Pipe Friction Factor [-]", Pipe.friction_factor)
PipeNetwork.track("Pipe Reynolds Number [-]", PipeFriction.reynolds_number)


# ---------------------------------------------------------------------------
# Transient Solve
# ---------------------------------------------------------------------------
# The solver uses a fixed timestep of 0.01 s and integrates to 5 s.
#
# The output file will be:
#
#     simple_pipe.h5
#
# For this simple example, the pipe starts from zero flow and approaches its
# steady-state flow rate as pressure force and friction loss come into balance.
Transient(PipeNetwork).solve(
    t_final=5,
    dt=0.01,
    verbose=True,
    filename="1simple_pipe",
)