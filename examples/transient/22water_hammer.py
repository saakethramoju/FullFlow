"""
GFSSP Example 15 Style Water-Hammer / Valve-Closure Model
=========================================================

This example builds a simplified FullFlow version of the GFSSP Example 15
water-hammer problem.

The model represents a long LOX feed line connected to a downstream valve that
closes rapidly. The pipe is divided into several lumped control volumes. Each
pipe segment has flow inertia, and each internal node has finite fluid storage.

The purpose of this example is to demonstrate:

    - lumped pipe inertia
    - distributed fluid storage using multiple Volume nodes
    - a sequenced valve-area closure
    - steady-state initialization before a transient valve closure
    - GFSSP-style branch/node discretization of a long pipe

Physical Layout
---------------

    LOX Source
    500 psia, 200 R
          |
          v
    +---------------------- 400 ft pipe ----------------------+
    |                                                         |
    v                                                         v

    Source
      |
      v
    Pipe Segment 1
      |
      v
    Pipe Node 1
      |
      v
    Pipe Segment 2
      |
      v
    Pipe Node 2
      |
      v
       ...
      |
      v
    Pipe Segment 5
      |
      v
    Pipe Node 5
      |
      v
    Closing Valve
      |
      v
    450 psia downstream boundary


GFSSP Reference Setup
---------------------

    Pipe length             = 400 ft
    Pipe diameter           = 0.25 in
    Initial source pressure = 500 psia
    Downstream pressure     = 450 psia
    Initial flow rate       = about 0.1 lbm/s
    Valve flow area         = 0.0491 in^2 initially
    Valve flow coefficient  = 0.6
    Valve closes in         = 0.1 s
    Pipe branch length      = 80 ft
    LOX wave speed          = 2462 ft/s
    Time step               = 0.02 s


Modeling Notes
--------------

GFSSP uses a finite-volume style network of branches and nodes. This example
uses the same idea:

    - Pipe segments carry momentum / flow inertia.
    - Volume nodes store fluid mass.
    - The outlet valve is algebraic and imposes a pressure-flow relation.
    - The valve area is sequenced as a function of time.

Each DarcyWeisbach pipe segment contributes a dynamic equation for mass flow.
Each Volume contributes a dynamic equation for mass storage.

During steady state, FullFlow drives all mass-flow derivatives and all node
mass derivatives to zero. This initializes the pipe at a consistent operating
point before the transient valve closure.

During transient, FullFlow integrates the pipe flow states and the node storage
states with an implicit time integrator.

This is still a lumped model. It is not a full method-of-characteristics water
hammer solver. Increasing the number of nodes gives a more distributed model of
the long line, but acoustic wave propagation accuracy depends on discretization,
time step, wave speed modeling, and the component equations being used.
"""

from fullflow import *
from thermoprop import *

import numpy as np


# -----------------------------------------------------------------------------
# Create network
# -----------------------------------------------------------------------------

WaterHammer = Network("Water Hammer")


# -----------------------------------------------------------------------------
# Unit conversions
#
# FullFlow uses SI units internally. The GFSSP reference problem is described
# mostly in English units, so the values are converted here once and then reused
# throughout the model.
# -----------------------------------------------------------------------------

psi_to_pa = 6894.757293168
ft_to_m = 0.3048
inch_to_m = 0.0254
in2_to_m2 = inch_to_m**2
lbm_to_kg = 0.45359237


# -----------------------------------------------------------------------------
# Boundary conditions
#
# The source and downstream pressures are fixed boundary states.
#
# The source pressure is 500 psia and the downstream boundary pressure is
# 450 psia, matching the GFSSP reference case.
#
# The LOX temperature is given as 200 R. Since Rankine uses the same increment
# size as Fahrenheit and Kelvin uses the same increment size as Celsius:
#
#     K = R * 5/9
# -----------------------------------------------------------------------------

source_pressure = State(500.0 * psi_to_pa)
downstream_pressure = State(450.0 * psi_to_pa)

lox_temperature = 200.0 * (5.0 / 9.0)


# -----------------------------------------------------------------------------
# Pipe geometry
#
# The total pipe is 400 ft long with a 0.25 inch inner diameter.
#
# The pipe is split into node_count equal lumped segments. Each segment gets a
# length dx, and each node gets a control volume equal to:
#
#     node_volume = pipe_area * dx
#
# This gives every internal node the fluid volume associated with one pipe
# segment.
# -----------------------------------------------------------------------------

pipe_length = 400.0 * ft_to_m
pipe_diameter = 0.25 * inch_to_m
pipe_area = (np.pi / 4.0) * pipe_diameter**2

node_count = 5
dx = pipe_length / node_count
node_volume = pipe_area * dx


# -----------------------------------------------------------------------------
# Time step / Courant number
#
# The wave speed is taken from the GFSSP reference problem.
#
# These Courant numbers are printed for reference only. They are useful when
# comparing to water-hammer style methods, but this example is still a lumped
# implicit network model rather than a pure method-of-characteristics model.
# -----------------------------------------------------------------------------

wave_speed = 2462.0 * ft_to_m
dt = 0.02
t_final = 5.0

courant_gfssp = 4.0 * dx / (wave_speed * dt)
courant_usual = wave_speed * dt / dx

print("GFSSP-style Courant number =", courant_gfssp)
print("Usual wave Courant number  =", courant_usual)


# -----------------------------------------------------------------------------
# Valve closing sequence
#
# GFSSP EX15VLV.DAT:
#
#     Time (s)    Flow Area (in^2)
#     0.00        0.0491
#     0.02        0.0164
#     0.04        0.00545
#     0.06        0.00182
#     0.08        0.00061
#     0.10        1.0e-16
#     100         1.0e-16
#
# The valve area is stored as a State because the Sequence component writes to
# it during the transient solve.
# -----------------------------------------------------------------------------

valve_area = State(0.0491 * in2_to_m2)

valve_times = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    100.0,
]

valve_areas = [
    0.0491 * in2_to_m2,
    0.0164 * in2_to_m2,
    0.00545 * in2_to_m2,
    0.00182 * in2_to_m2,
    0.00061 * in2_to_m2,
    1.0e-16 * in2_to_m2,
    1.0e-16 * in2_to_m2,
]


ValveAreaSequence = Sequence(
    "Valve Area Sequence",
    WaterHammer,
    target=valve_area,
    times=valve_times,
    values=valve_areas,
)


# -----------------------------------------------------------------------------
# Source fluid lookup
#
# The source is liquid oxygen at the fixed source pressure and temperature.
# ThermoProp supplies density, viscosity, enthalpy, and other fluid properties.
# -----------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source LOX",
    WaterHammer,
    Fluid,
    "oxygen",
    pressure=source_pressure,
    temperature=lox_temperature,
)


# -----------------------------------------------------------------------------
# Lists used to build the repeated pipe/node chain
#
# node_pressures:
#     pressure State for each internal pipe node
#
# node_fluids:
#     ThermoProp Fluid lookup for each internal pipe node
#
# pipe_flows:
#     mass-flow State for each pipe segment
#
# pipe_segments:
#     DarcyWeisbach components for each pipe segment
#
# pipe_nodes:
#     Volume components for each internal node
# -----------------------------------------------------------------------------

node_pressures = []
node_fluids = []
pipe_flows = []
pipe_segments = []
pipe_nodes = []


# -----------------------------------------------------------------------------
# Initial pressure guesses
#
# The internal node pressures are initialized with a linear interpolation between
# the source pressure and downstream pressure.
#
# These are only initial guesses. The steady-state solver adjusts them until the
# pipe and node equations are satisfied.
# -----------------------------------------------------------------------------

for i in range(node_count):
    pressure_guess = source_pressure.value - (source_pressure.value - downstream_pressure.value) * (i + 1) / (node_count + 1)
    node_pressures.append(State(pressure_guess))


# -----------------------------------------------------------------------------
# Node fluid lookups
#
# Each pipe node gets its own fluid lookup. The pressure is the node pressure
# state, and the temperature is held at the initial LOX temperature.
#
# This example only solves mass storage in the Volume nodes, so temperature and
# energy are not dynamic here.
# -----------------------------------------------------------------------------

for i in range(node_count):
    fluid = Lookup(
        f"Node {i + 1} LOX",
        WaterHammer,
        Fluid,
        "oxygen",
        pressure=node_pressures[i],
        temperature=lox_temperature,
    )

    node_fluids.append(fluid)


# -----------------------------------------------------------------------------
# Initial pipe flow guesses
#
# GFSSP reports an initial flow rate of about 0.1 lbm/s. Each pipe segment gets
# this same initial mass-flow guess.
# -----------------------------------------------------------------------------

for i in range(node_count):
    pipe_flows.append(State(0.1 * lbm_to_kg))


# -----------------------------------------------------------------------------
# Pipe segments
#
# Each pipe segment is a dynamic Darcy-Weisbach branch.
#
# Segment 1 connects the fixed source pressure to Node 1.
# Segment i connects Node i-1 to Node i.
#
# Density is taken from the upstream side of each segment. This is a simple
# lumped approximation and keeps the model close to a node/branch network style.
# -----------------------------------------------------------------------------

for i in range(node_count):
    if i == 0:
        upstream_pressure = source_pressure
        downstream_pressure_i = node_pressures[i]
        density = SourceFluid.density
    else:
        upstream_pressure = node_pressures[i - 1]
        downstream_pressure_i = node_pressures[i]
        density = node_fluids[i - 1].density

    segment = DarcyWeisbach(
        f"Pipe Segment {i + 1}",
        WaterHammer,
        mass_flow=pipe_flows[i],
        upstream_pressure=upstream_pressure,
        downstream_pressure=downstream_pressure_i,
        length=dx,
        hydraulic_diameter=pipe_diameter,
        density=density,
        cross_sectional_area=pipe_area,
        friction_factor=0.02,
    )

    pipe_segments.append(segment)


# -----------------------------------------------------------------------------
# Outlet valve
#
# The valve is an algebraic discharge-coefficient restriction.
#
# The sequenced valve_area is used as the cross-sectional area. As the sequence
# closes the area, the outlet flow decreases.
#
# No length is provided here. That is intentional: the valve itself is treated as
# a restriction, not as an inertial pipe slug. The pipe inertia is already in
# the DarcyWeisbach pipe segments upstream.
# -----------------------------------------------------------------------------

OutletValve = DischargeCoefficient(
    "Outlet Valve",
    WaterHammer,
    upstream_pressure=node_pressures[-1],
    downstream_pressure=downstream_pressure,
    density=node_fluids[-1].density,
    discharge_coefficient=0.6,
    cross_sectional_area=valve_area,
)


# -----------------------------------------------------------------------------
# Pipe control volumes
#
# Each internal node stores mass in a finite control volume.
#
# For node i:
#
#     mass_flow_in  = pipe_flows[i]
#     mass_flow_out = pipe_flows[i + 1]
#
# except for the final node, whose mass_flow_out is the outlet valve flow.
#
# In steady state, each Volume drives:
#
#     mass_flow_in - mass_flow_out = 0
#
# In transient, each Volume integrates mass storage:
#
#     d(mass) / dt = mass_flow_in - mass_flow_out
# -----------------------------------------------------------------------------

for i in range(node_count):
    if i == node_count - 1:
        mass_flow_out = OutletValve.mass_flow
    else:
        mass_flow_out = pipe_flows[i + 1]

    node = Volume(
        f"Pipe Node {i + 1}",
        WaterHammer,
        pressure=node_pressures[i],
        volume=node_volume,
        density=node_fluids[i].density,
        mass_flow_in=pipe_flows[i],
        mass_flow_out=mass_flow_out,
    )

    pipe_nodes.append(node)


# -----------------------------------------------------------------------------
# Solve
#
# The steady-state solve initializes the network before valve closure.
# -----------------------------------------------------------------------------

filename = "22water_hammer"


SteadyState(WaterHammer).solve(
    verbose=True,
    filename=filename,
)


Transient(WaterHammer).solve(
    dt=dt,
    t_final=t_final,
    filename=filename,
    verbose=False,
    statistics=True,
    rtol=1.0e-5,
)
