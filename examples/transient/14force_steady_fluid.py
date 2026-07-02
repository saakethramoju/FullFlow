"""
Simple forced-steady fluid time-sweep example.

Physical layout
---------------

    Source Pressure Schedule
          |
          v
    Pipe 1 with inertia and friction
          |
          v
    Node 1 with mass/pressure storage
          |
          v
    Pipe 2 with inertia and friction
          |
          v
    Node 2 with mass/pressure storage
          |
          v
    Pipe 3 with inertia and friction
          |
          v
    Fixed Outlet Pressure

This example solves the same network two ways:

1. Fully dynamic
   - Pipe mass flows are integrated.
   - Node pressures are integrated through mass storage.

2. Force-steady everything except the nodes
   - SteadyState.solve(dt=..., t_final=...) tells FullFlow to force every
     dynamic component to steady state at each time point.
   - exceptions=[Node1, Node2] keeps the node pressure/mass storage dynamics
     active.
   - Therefore the pipe mass-flow inertia is removed, but the node pressure
     storage is still dynamic.

No custom components are used. The pipe friction factors are constant to keep
this example focused only on forced-steady behavior.
"""

import math
import matplotlib.pyplot as plt

from fullflow import *


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------

source_pressure_initial = 300000.0  # Pa
source_pressure_final = 500000.0    # Pa
outlet_pressure = 250000.0          # Pa

reference_pressure = 300000.0       # Pa
reference_density = 1000.0          # kg/m^3
compressibility = 5.0e-9            # 1/Pa, simple liquid compressibility

pipe_diameter = 0.0254              # m
pipe_area = math.pi / 4.0 * pipe_diameter**2
pipe_length = 3.0                   # m
pipe_friction_factor = 0.03

node_volume = 0.02                  # m^3


# -----------------------------------------------------------------------------
# Input schedule
# -----------------------------------------------------------------------------

def source_pressure_command(t):
    """Step the source pressure after the steady-state initialization."""
    if t < 0.2:
        return source_pressure_initial

    return source_pressure_final


# -----------------------------------------------------------------------------
# Network builder
# -----------------------------------------------------------------------------

def build_network(name):
    """Build one copy of the same two-node, three-pipe fluid network."""

    PipeNetwork = Network(name)

    # Boundary pressures.
    SourcePressure = State(source_pressure_initial)
    OutletPressure = State(outlet_pressure)

    Sequence(
        "Source Pressure Schedule",
        PipeNetwork,
        target=SourcePressure,
        function=source_pressure_command,
    )

    # Node pressures are solve variables owned by the Volume dynamics.
    Node1Pressure = State(285000.0)
    Node2Pressure = State(270000.0)

    # Simple pressure-dependent densities. This keeps the example
    # self-contained while still giving the nodes real pressure/mass storage.
    SourceDensity = reference_density * (1.0 + compressibility * (SourcePressure - reference_pressure))
    Node1Density = reference_density * (1.0 + compressibility * (Node1Pressure - reference_pressure))
    Node2Density = reference_density * (1.0 + compressibility * (Node2Pressure - reference_pressure))

    # Shared pipe mass-flow states.
    Pipe1MassFlow = State(0.5)
    Pipe2MassFlow = State(0.5)
    Pipe3MassFlow = State(0.5)

    # Frictional lines with inertia.
    Pipe1 = DarcyWeisbach(
        "Pipe 1",
        PipeNetwork,
        mass_flow=Pipe1MassFlow,
        upstream_pressure=SourcePressure,
        downstream_pressure=Node1Pressure,
        length=pipe_length,
        hydraulic_diameter=pipe_diameter,
        density=SourceDensity,
        cross_sectional_area=pipe_area,
        friction_factor=pipe_friction_factor,
    )

    Pipe2 = DarcyWeisbach(
        "Pipe 2",
        PipeNetwork,
        mass_flow=Pipe2MassFlow,
        upstream_pressure=Node1Pressure,
        downstream_pressure=Node2Pressure,
        length=pipe_length,
        hydraulic_diameter=pipe_diameter,
        density=Node1Density,
        cross_sectional_area=pipe_area,
        friction_factor=pipe_friction_factor,
    )

    Pipe3 = DarcyWeisbach(
        "Pipe 3",
        PipeNetwork,
        mass_flow=Pipe3MassFlow,
        upstream_pressure=Node2Pressure,
        downstream_pressure=OutletPressure,
        length=pipe_length,
        hydraulic_diameter=pipe_diameter,
        density=Node2Density,
        cross_sectional_area=pipe_area,
        friction_factor=pipe_friction_factor,
    )

    # Nodes with pressure/mass storage.
    Node1 = Volume(
        "Node 1",
        PipeNetwork,
        pressure=Node1Pressure,
        volume=node_volume,
        density=Node1Density,
        mass_flow_in=Pipe1MassFlow,
        mass_flow_out=Pipe2MassFlow,
    )

    Node2 = Volume(
        "Node 2",
        PipeNetwork,
        pressure=Node2Pressure,
        volume=node_volume,
        density=Node2Density,
        mass_flow_in=Pipe2MassFlow,
        mass_flow_out=Pipe3MassFlow,
    )

    PipeNetwork.track("Source Pressure [Pa]", SourcePressure)
    PipeNetwork.track("Node 1 Pressure [Pa]", Node1Pressure)
    PipeNetwork.track("Node 2 Pressure [Pa]", Node2Pressure)
    PipeNetwork.track("Pipe 1 Mass Flow [kg/s]", Pipe1MassFlow)
    PipeNetwork.track("Pipe 2 Mass Flow [kg/s]", Pipe2MassFlow)
    PipeNetwork.track("Pipe 3 Mass Flow [kg/s]", Pipe3MassFlow)

    return PipeNetwork, Node1, Node2, Pipe1, Pipe2, Pipe3


# -----------------------------------------------------------------------------
# Small plotting helper
# -----------------------------------------------------------------------------

def history_track(solver, track_name):
    """Read one tracked output from a Transient solver history."""
    rows = [
        row
        for row in solver.history.track_records
        if row["attribute"] == track_name
    ]

    times = [row["time"] for row in rows]
    values = [row["value"] for row in rows]

    return times, values


# -----------------------------------------------------------------------------
# Case 1: fully dynamic pipes and nodes
# -----------------------------------------------------------------------------

DynamicNetwork, DynamicNode1, DynamicNode2, DynamicPipe1, DynamicPipe2, DynamicPipe3 = build_network(
    "Fully Dynamic Fluid Network"
)

SteadyState(DynamicNetwork).solve(filename="force_steady_fully_dynamic")

DynamicSolver = Transient(DynamicNetwork)
DynamicSolver.solve(
    dt=0.01,
    t_final=2.0,
    save_dt=0.01,
    filename="force_steady_fully_dynamic",
)


# -----------------------------------------------------------------------------
# Case 2: force-steady all dynamics except the node storage dynamics
# -----------------------------------------------------------------------------

ForceSteadyNetwork, ForceSteadyNode1, ForceSteadyNode2, ForceSteadyPipe1, ForceSteadyPipe2, ForceSteadyPipe3 = build_network(
    "Force-Steady Pipe Fluid Network"
)

SteadyState(ForceSteadyNetwork).solve(filename="14force_steady_fluid")

ForceSteadySolver = SteadyState(ForceSteadyNetwork)
ForceSteadySolver.solve(
    dt=0.01,
    t_final=2.0,
    save_dt=0.01,
    filename="14force_steady_fluid",
    exceptions=[
        ForceSteadyNode1,
        ForceSteadyNode2,
    ],
)