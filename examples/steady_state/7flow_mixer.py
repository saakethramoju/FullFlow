"""
Dye Mixer with Darcy-Weisbach Inlet and Outlet Lines
===================================================

This example models two water streams carrying different dye concentrations.
The streams flow through separate inlet lines, mix in a common node, and leave
through a shared outlet line.

The purpose of this example is to show how the Composition component can be
used with a single transported scalar, while the mass flows are solved by
pressure-driven Darcy-Weisbach line components instead of being prescribed.

Physical Layout
---------------

        Source A                         Source B
    dye = 0.5                         dye = 0.2
    P = 300 kPa                       P = 250 kPa
        |                                 |
        |                                 |
        v                                 v
 +----------------+              +----------------+
 | Source A Line  |              | Source B Line  |
 +----------------+              +----------------+
        |                                 |
        +---------------+-----------------+
                        |
                        v
                 +-------------+
                 | Mixer Node  |
                 +-------------+
                        |
                        v
                +----------------+
                | Outlet Line    |
                +----------------+
                        |
                        v
                  Mixed Output
                  P = 100 kPa

Mass Flow Model
---------------

The three DarcyWeisbach components solve the line mass flows from pressure drop,
tube geometry, water density, and a reasonable assumed Darcy friction factor.

The mixer node pressure is also solved. The Volume component enforces:

    SourceA.mass_flow + SourceB.mass_flow = Mixed.mass_flow

Composition Model
-----------------

Only one scalar is tracked: dye concentration.

The Composition component enforces:

    mdot_A * dye_A + mdot_B * dye_B = mdot_mixed * dye_mixed

The outlet uses the None shortcut:

    outlets=[(Mixed.mass_flow, None)]

Since solve={"dye": Mixed.dye_percent}, None means the outlet composition is:

    {"dye": Mixed.dye_percent}

This keeps the single-scalar mixer example compact while still using the same
Composition machinery that works for larger species dictionaries.
"""

from fullflow import *


MixerNetwork = Network("Mixer Network")


# -----------------------------------------------------------------------------
# Water properties and line assumptions
# -----------------------------------------------------------------------------
# This example uses constant water properties to keep the focus on the network
# structure. A Darcy friction factor of 0.02 is a reasonable simple turbulent-flow
# starting point for a smooth pipe example.
water_density = State(997.0)
friction_factor = 0.02


# -----------------------------------------------------------------------------
# Source A
# -----------------------------------------------------------------------------
# Source A is a fixed-pressure inlet stream with a high dye concentration.
SourceA = Component("Source A", MixerNetwork)
SourceA.pressure = State(300_000.0)
SourceA.mass_flow = State(8.5)
SourceA.dye_concentration = {"dye": 0.5}


# -----------------------------------------------------------------------------
# Source B
# -----------------------------------------------------------------------------
# Source B is another fixed-pressure inlet stream with a lower dye concentration.
SourceB = Component("Source B", MixerNetwork)
SourceB.pressure = State(250_000.0)
SourceB.mass_flow = State(12.5)
SourceB.dye_concentration = {"dye": 0.2}


# -----------------------------------------------------------------------------
# Mixed outlet
# -----------------------------------------------------------------------------
# The mixed outlet pressure is fixed. Its mass flow is solved by the outlet line,
# and the mixer node mass balance forces it to equal the sum of the two inlet
# mass flows.
Mixed = Component("Mixed Output", MixerNetwork)
Mixed.pressure = State(100_000.0)
Mixed.mass_flow = State(21.0)
Mixed.dye_percent = State(0.3)
Mixed.dye_concentration = {"dye": Mixed.dye_percent}


# -----------------------------------------------------------------------------
# Mixer node
# -----------------------------------------------------------------------------
# The mixer node is a real lumped storage volume. In steady state, FullFlow
# drives its mass derivative to zero:
#
#     mass_dot = SourceA.mass_flow + SourceB.mass_flow - Mixed.mass_flow = 0
#
# The solver varies the node pressure until that derivative is zero.
MixerNode = Volume(
    "Mixer Node",
    MixerNetwork,
    pressure=State(150_000.0),
    volume=1.0,
    density=water_density,
    mass_flow_in=SourceA.mass_flow + SourceB.mass_flow,
    mass_flow_out=Mixed.mass_flow,
)


# -----------------------------------------------------------------------------
# Source A inlet line
# -----------------------------------------------------------------------------
# This line determines SourceA.mass_flow from the pressure drop between Source A
# and the mixer node.
SourceALine = DarcyWeisbach(
    "Source A Line",
    MixerNetwork,
    mass_flow=SourceA.mass_flow,
    upstream_pressure=SourceA.pressure,
    downstream_pressure=MixerNode.pressure,
    length=3.0,
    hydraulic_diameter=0.030,
    density=water_density,
    friction_factor=friction_factor,
)


# -----------------------------------------------------------------------------
# Source B inlet line
# -----------------------------------------------------------------------------
# This line determines SourceB.mass_flow from the pressure drop between Source B
# and the mixer node.
SourceBLine = DarcyWeisbach(
    "Source B Line",
    MixerNetwork,
    mass_flow=SourceB.mass_flow,
    upstream_pressure=SourceB.pressure,
    downstream_pressure=MixerNode.pressure,
    length=4.0,
    hydraulic_diameter=0.040,
    density=water_density,
    friction_factor=friction_factor,
)


# -----------------------------------------------------------------------------
# Mixed outlet line
# -----------------------------------------------------------------------------
# This line determines the outlet mass flow from the pressure drop between the
# mixer node and the downstream outlet.
OutletLine = DarcyWeisbach(
    "Mixed Outlet Line",
    MixerNetwork,
    mass_flow=Mixed.mass_flow,
    upstream_pressure=MixerNode.pressure,
    downstream_pressure=Mixed.pressure,
    length=2.0,
    hydraulic_diameter=0.050,
    density=water_density,
    friction_factor=friction_factor,
)


# -----------------------------------------------------------------------------
# Composition balance
# -----------------------------------------------------------------------------
# The inlet compositions are fixed. The mixed outlet dye concentration is unknown.
#
# Because only one scalar is being tracked, the solve dictionary has one entry:
#
#     {"dye": Mixed.dye_percent}
#
# The outlet composition is passed as None, which tells Composition to use the
# solve dictionary for that outlet.
Mixer = Composition(
    "Mixer",
    MixerNetwork,
    inlets=[
        (SourceA.mass_flow, SourceA.dye_concentration),
        (SourceB.mass_flow, SourceB.dye_concentration),
    ],
    outlets=[
        (Mixed.mass_flow, None),
    ],
    solve={"dye": Mixed.dye_percent},
)


SteadyState(MixerNetwork).solve(verbose=True)