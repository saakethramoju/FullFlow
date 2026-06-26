"""
Transient Node Inertia Example
==============================

This example models a simple feed-line, lumped-node, and valve-opening
transient.

The goal is to demonstrate how a small internal node volume responds when an
upstream inertial pipe feeds an outlet valve that opens with time.

Physical Layout
---------------

    Pressurized RP-1 Source
        P = 300 kPa
        T = 300 K
            |
            |  Pipe 1
            |  - Darcy-Weisbach pressure loss
            |  - finite flow inertia through length
            |  - friction factor updated by Churchill correlation
            v
    Lumped Node / Manifold
        - small finite volume
        - pressure is solved from mass conservation
        - enthalpy is solved from energy conservation
        - density/internal energy come from ThermoProp lookup
            |
            |  Pipe 2 / Valve
            |  - algebraic discharge-coefficient relation
            |  - no length, so no valve inertia
            |  - Cd opens smoothly from nearly closed to fully open
            v
    Atmosphere
        P = 101325 Pa


Modeling Notes
--------------

Pipe 1 is dynamic because it has length. Its mass flow cannot change
instantaneously. The solver integrates its mass-flow inertia during transient
and drives mass_flow_dot to zero during steady state.

The node is dynamic because it stores fluid mass and internal energy. The
solver varies pressure and enthalpy while conserving the extensive quantities
mass and total internal energy.

Pipe 2 is intentionally algebraic. A valve/orifice should usually be modeled as
a restriction, not as an inertial line. If a length is added directly to a valve
component, a nearly closed valve becomes a very stiff and physically awkward
inertial branch.

The steady-state solve initializes the system at the closed-valve rest
condition. The transient solve then opens the valve smoothly and lets the node
pressure, inlet flow, and outlet flow evolve with time.
"""

import math

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Create network
# -----------------------------------------------------------------------------

PipeNetwork = Network("Pipe Network")


# -----------------------------------------------------------------------------
# Source fluid
#
# The source is a fixed upstream RP-1 state. Its pressure and temperature are
# prescribed, and ThermoProp provides density, viscosity, enthalpy, etc.
# -----------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source Fluid",
    PipeNetwork,
    Fluid,
    "rp-1",
    pressure=3e5,
    temperature=300,
)


# -----------------------------------------------------------------------------
# Node fluid
#
# The node uses the same fluid as the source.
#
# pressure=101325 and temperature=300 are initial guesses. During the steady and
# transient solves, pressure is changed by the Volume mass equation and enthalpy
# is changed by the Volume energy equation.
#
# priority=("enthalpy", "temperature") means:
#   1. start from the easy pressure-temperature guess
#   2. seed enthalpy from that state
#   3. then use pressure-enthalpy as the active thermodynamic state
# -----------------------------------------------------------------------------

NodeFluid = Lookup(
    "Node Fluid",
    PipeNetwork,
    Fluid,
    SourceFluid.fluid,
    pressure=101325,
    temperature=300,
    priority=("enthalpy", "temperature"),
)


# -----------------------------------------------------------------------------
# Geometry
#
# The pipe diameter is given as 2 inches and converted to meters. The same area
# is reused for the feed pipe, node volume estimate, and outlet valve.
# -----------------------------------------------------------------------------

pipe_diameter = 2.0 / 39.37
pipe_area = (math.pi / 4) * pipe_diameter**2


# -----------------------------------------------------------------------------
# Pipe 1: upstream inertial feed line
#
# DarcyWeisbach is dynamic because it has a finite length. The state
# Pipe1.mass_flow is integrated in transient and trimmed in steady state.
# -----------------------------------------------------------------------------

Pipe1 = DarcyWeisbach(
    "Pipe 1",
    PipeNetwork,
    mass_flow=0,
    upstream_pressure=SourceFluid.pressure,
    downstream_pressure=NodeFluid.pressure,
    length=5,
    hydraulic_diameter=pipe_diameter,
    density=SourceFluid.density,
    cross_sectional_area=pipe_area,
    friction_factor=2e-4,
)


# -----------------------------------------------------------------------------
# Pipe 1 friction update
#
# Churchill updates Pipe1.friction_factor from the current Reynolds number.
# It is an explicit calculator component: it evaluates every pass, but it does
# not add a new dynamics or balance equation.
# -----------------------------------------------------------------------------

Pipe1Friction = Churchill(
    "Pipe 1 Friction",
    PipeNetwork,
    mass_flow=Pipe1.mass_flow,
    friction_factor=Pipe1.friction_factor,
    hydraulic_diameter=Pipe1.hydraulic_diameter,
    dynamic_viscosity=SourceFluid.dynamic_viscosity,
    cross_sectional_area=Pipe1.cross_sectional_area,
    roughness=1e-5,
)


# -----------------------------------------------------------------------------
# Shared outlet mass-flow state
#
# Pipe 2 writes this state and the node reads it as mass_flow_out.
# -----------------------------------------------------------------------------

pipe2_mass_flow = State(0)


# -----------------------------------------------------------------------------
# Lumped node / small manifold volume
#
# The node volume is estimated as one pipe diameter of volume:
#
#     V = A * D
#
# The Volume component owns:
#   - mass conservation
#   - optional energy conservation
#
# Since enthalpy is provided, the default energy_variable="enthalpy" is used.
# That means the solver uses NodeFluid.enthalpy as the energy unknown while
# integrating total internal energy.
# -----------------------------------------------------------------------------

Node = Volume(
    "Node",
    network=PipeNetwork,
    volume=pipe_area * pipe_diameter,
    pressure=NodeFluid.pressure,
    enthalpy=NodeFluid.enthalpy,
    density=NodeFluid.density,
    temperature=NodeFluid.temperature,
    internal_energy=NodeFluid.internal_energy,
    mass_flow_in=Pipe1.mass_flow,
    mass_flow_out=pipe2_mass_flow,
    total_enthalpy_in=SourceFluid.enthalpy,
)


# -----------------------------------------------------------------------------
# Smooth valve-opening sequence
#
# The valve remains nearly closed until valve_delay, then opens smoothly over
# valve_open_time using a smoothstep ramp.
#
# A tiny Cd value is used instead of exactly zero so the component remains
# numerically well-defined, but the DischargeCoefficient component still treats
# tiny CdA as essentially closed in algebraic mode.
# -----------------------------------------------------------------------------

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


valve_delay = 0.10
valve_open_time = 1.00
n_points = 101

valve_times = [0.0, valve_delay]
valve_cds = [1e-12, 1e-12]

for i in range(1, n_points + 1):
    x = i / n_points
    t = valve_delay + valve_open_time * x

    valve_times.append(t)
    valve_cds.append(1e-12 + (1.0 - 1e-12) * smoothstep(x))

valve_times.append(100.0)
valve_cds.append(1.0)


ValveCdSequence = Sequence(
    "Valve Cd Sequence",
    PipeNetwork,
    times=valve_times,
    values=valve_cds,
)


# -----------------------------------------------------------------------------
# Pipe 2: outlet valve / discharge coefficient restriction
#
# This component is intentionally used without length.
#
# Without length:
#   - DischargeCoefficient is algebraic
#   - mass_flow is written directly from pressure drop and CdA
#   - this is appropriate for a valve/orifice-like restriction
#
# With length:
#   - DischargeCoefficient becomes an inertial branch
#   - this can be useful for open line inertia
#   - it is usually not appropriate for a fully closed valve
# -----------------------------------------------------------------------------

Pipe2 = DischargeCoefficient(
    "Pipe 2",
    PipeNetwork,
    upstream_pressure=Node.pressure,
    downstream_pressure=101325,
    density=NodeFluid.density,
    discharge_coefficient=ValveCdSequence.target,
    cross_sectional_area=pipe_area,
    mass_flow=pipe2_mass_flow,
)


# -----------------------------------------------------------------------------
# Solve and export
#
# Both the steady-state initialization and transient result are written to the
# same HDF5 file. The steady solve initializes the closed-valve rest condition;
# the transient solve then opens the valve according to ValveCdSequence.
# -----------------------------------------------------------------------------

filename = "node_inertia"

SteadyState(PipeNetwork).solve(
    verbose=True,
    filename=filename,
)

Transient(PipeNetwork).solve(
    t_final=5,
    dt=0.01,
    verbose=True,
    filename=filename,
)