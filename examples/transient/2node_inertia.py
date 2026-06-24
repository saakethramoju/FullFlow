import math

from fullflow import *
from thermoprop import *


PipeNetwork = Network("Pipe Network")



SourceFluid = Lookup(
    "Source Fluid",
    PipeNetwork,
    Propellant,
    "rp-1",
    pressure=3e5,
    temperature=300,
)


NodeFluid = Lookup(
    "Node Fluid",
    PipeNetwork,
    Propellant,
    SourceFluid.propellant,
    pressure = 101325,
    temperature=300
)


Pipe1 = DarcyWeisbach(
    "Pipe 1",
    PipeNetwork,
    mass_flow=0,
    upstream_pressure=SourceFluid.pressure,
    downstream_pressure=NodeFluid.pressure,
    length=5,
    hydraulic_diameter=2.0 / 39.37,
    density=SourceFluid.density,
    cross_sectional_area=(math.pi/4) * (2.0 / 39.37)**2,
    friction_factor=2e-4,
)

Pipe1Friction = Churchill(
    "Pipe 1 Friction",
    PipeNetwork,
    mass_flow=Pipe1.mass_flow,
    friction_factor=Pipe1.friction_factor,
    hydraulic_diameter=Pipe1.hydraulic_diameter,
    dynamic_viscosity=SourceFluid.dynamic_viscosity,
    cross_sectional_area=Pipe1.cross_sectional_area,
    roughness=1e-5
)

pipe2_mass_flow = State(0)

Node = Volume(
    "Node",
    network=PipeNetwork,
    volume= (math.pi/4) * (2.0 / 39.37)**2 * (2.0/39.37),
    pressure=NodeFluid.pressure,
    density=NodeFluid.density,
    mass_flow_in=Pipe1.mass_flow,
    mass_flow_out=pipe2_mass_flow
)

valve_times = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    100.0,
]


valve_cds = [
    1e-12,
    0.001,
    0.01,
    0.35,
    0.6,
    1.0,
    1.0
]

ValveCdSchedule = Schedule(
    "Valve Cd Schedule",
    PipeNetwork,
    #target=valve_area,
    times=valve_times,
    values=valve_cds,
)


Pipe2 = DischargeCoefficient(
    "Pipe 2",
    PipeNetwork,
    upstream_pressure=Node.pressure,
    downstream_pressure=101325,
    density=NodeFluid.density,
    discharge_coefficient=ValveCdSchedule.target,
    cross_sectional_area=(math.pi/4) * (2.0 / 39.37)**2,
    length=4.5, # without length, DischargeCoefficient becomes purely algebraic
    mass_flow=pipe2_mass_flow
)


filename = "node_inertia" # in an h5 file, both Steady and Transient results get stored under the same group for a given Network

SteadyState(PipeNetwork).solve(
    verbose=True,
    #filename=filename
)



'''
Transient(PipeNetwork).solve(
    t_final=5,
    dt=0.01,
    verbose=True,
    filename=filename
)
'''