from fullflow import *
from thermoprop import *

import numpy as np


"""
GFSSP Example 15 style water-hammer / valve-closure model.

Physical layout
---------------

    LOX Source
    500 psia, 200 R
          |
          v
    +---------------------- 400 ft pipe ----------------------+
    |                                                         |
    v                                                         v

    Source -> Pipe 1 -> Node 1 -> Pipe 2 -> Node 2 -> ...
           -> Pipe 5 -> Node 5 -> Closing Valve -> 450 psia boundary


GFSSP reference setup
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
"""


Test = Network("GFSSP Example 15 Valve Closure")


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

psi_to_pa = 6894.757293168
ft_to_m = 0.3048
inch_to_m = 0.0254
in2_to_m2 = inch_to_m**2
lbm_to_kg = 0.45359237


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

source_pressure = State(500.0 * psi_to_pa)
downstream_pressure = State(450.0 * psi_to_pa)

lox_temperature = 200.0 * (5.0 / 9.0)       # Rankine to Kelvin


# ---------------------------------------------------------------------------
# Pipe geometry
# ---------------------------------------------------------------------------

pipe_length = 400.0 * ft_to_m
pipe_diameter = 0.25 * inch_to_m
pipe_area = (np.pi / 4.0) * pipe_diameter**2

node_count = 5
dx = pipe_length / node_count
node_volume = pipe_area * dx


# ---------------------------------------------------------------------------
# Time step / Courant number
# ---------------------------------------------------------------------------

wave_speed = 2462.0 * ft_to_m
dt = 0.02
t_final = 5.0

courant_gfssp = 4.0 * dx / (wave_speed * dt)
courant_usual = wave_speed * dt / dx

print("GFSSP-style Courant number =", courant_gfssp)
print("Usual wave Courant number  =", courant_usual)


# ---------------------------------------------------------------------------
# Valve closing schedule
# ---------------------------------------------------------------------------
# GFSSP EX15VLV.DAT:
#
# Time (s)    Flow Area (in^2)
# 0.00        0.0491
# 0.02        0.0164
# 0.04        0.00545
# 0.06        0.00182
# 0.08        0.00061
# 0.10        1.0e-16
# 100         1.0e-16

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


ValveAreaSchedule = Schedule(
    "Valve Area Schedule",
    Test,
    target=valve_area,
    times=valve_times,
    values=valve_areas,
)


# ---------------------------------------------------------------------------
# Fluid lookups
# ---------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source LOX",
    Test,
    Fluid,
    "oxygen",
    pressure=source_pressure,
    temperature=lox_temperature,
)


node_pressures = []
node_fluids = []
pipe_flows = []
pipe_segments = []
pipe_nodes = []


# ---------------------------------------------------------------------------
# Initial pressure guesses
# ---------------------------------------------------------------------------

for i in range(node_count):
    pressure_guess = source_pressure.value - (source_pressure.value - downstream_pressure.value) * (i + 1) / (node_count + 1)
    node_pressures.append(State(pressure_guess))


for i in range(node_count):
    fluid = Lookup(
        f"Node {i + 1} LOX",
        Test,
        Fluid,
        "oxygen",
        pressure=node_pressures[i],
        temperature=lox_temperature,
    )

    node_fluids.append(fluid)


# Initial flow guess: 0.1 lbm/s
for i in range(node_count):
    pipe_flows.append(State(0.1 * lbm_to_kg))


# ---------------------------------------------------------------------------
# Pipe segments
# ---------------------------------------------------------------------------

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
        Test,
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


# ---------------------------------------------------------------------------
# Outlet valve
# ---------------------------------------------------------------------------

OutletValve = DischargeCoefficient(
    "Outlet Valve",
    Test,
    upstream_pressure=node_pressures[-1],
    downstream_pressure=downstream_pressure,
    density=node_fluids[-1].density,
    discharge_coefficient=0.6,
    cross_sectional_area=valve_area,
)


# ---------------------------------------------------------------------------
# Pipe control volumes
# ---------------------------------------------------------------------------

for i in range(node_count):
    if i == node_count - 1:
        mass_flow_out = OutletValve.mass_flow
    else:
        mass_flow_out = pipe_flows[i + 1]

    node = Volume(
        f"Pipe Node {i + 1}",
        Test,
        pressure=node_pressures[i],
        volume=node_volume,
        density=node_fluids[i].density,
        mass_flow_in=pipe_flows[i],
        mass_flow_out=mass_flow_out,
    )

    pipe_nodes.append(node)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

filename = "gfssp_example_15_valve_closure"


SteadyState(Test).solve(
    verbose=True,
    filename=filename,
)


Transient(Test).solve(
    dt=dt,
    t_final=t_final,
    filename=filename,
    verbose=False,
    statistics=True,
    rtol=1.0e-5,
)