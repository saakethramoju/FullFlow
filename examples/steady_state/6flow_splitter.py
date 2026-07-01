"""
H2O Separator with Flow Tubes and Composition Conservation
==========================================================

This example demonstrates a steady-state gas separator network using FullFlow
and ThermoProp. A reacting H2/O2 equilibrium gas mixture enters a separator,
a pure H2O product stream is removed, and the remaining gas leaves through an
exhaust outlet.

The example is intentionally small, but it combines several important FullFlow
features:

    1. ThermoProp + Lookup for equilibrium gas properties.
    2. FlowTube components for pressure-driven inlet and outlet flow.
    3. Volume as a separator node with a mass balance.
    4. Composition for species mass-fraction conservation.
    5. State.from_iterable(...) for turning a known dictionary into an unknown
       dictionary whose values can be solver variables.

Physical Layout
---------------

                          chamber gas
                     H2/O2 equilibrium mixture
                              |
                              v
    +---------------------------------------------------+
    |                                                   |
    |              Chamber to Separator Tube             |
    |                                                   |
    +---------------------------------------------------+
                              |
                              v
                     +-----------------+
                     |                 |
                     | Separator Node  |
                     |                 |
                     +-----------------+
                        |           |
                        |           |
                        v           v
             +----------------+   +----------------+
             |                |   |                |
             | H2O Product    |   | Exhaust Gas    |
             | Outlet Tube    |   | Outlet Tube    |
             |                |   |                |
             +----------------+   +----------------+
                        |           |
                        v           v
                  pure H2O      remaining gas mixture

Mass Flow Model
---------------

The separator node enforces total mass conservation:

    chamber mass flow = product mass flow + exhaust mass flow

The three FlowTube components solve the mass flows from pressure drops:

    chamber pressure   -> separator pressure
    separator pressure -> product outlet pressure
    separator pressure -> exhaust outlet pressure

A simple Darcy friction factor is provided for each tube. The values are not
intended to represent a specific design; they simply make the pressure-flow
relationships well posed for this user example.

Composition Model
-----------------

The chamber composition is known from ThermoProp equilibrium.

The product is prescribed as pure H2O:

    Product.composition = {"H2O": 1.0}

The exhaust composition is unknown. Its initial species values are copied from
the chamber composition and converted into State objects with
State.from_iterable(...).

Because mass fractions should sum to one, one species is treated as the
dependent remainder. Here H2O is used as the remainder:

    exhaust H2O = 1 - sum(other exhaust species)

The Composition component receives the full inlet and outlet dictionaries, but
it solves only the independent non-H2O species. H2O conservation is implied by
the total mass balance plus the exhaust sum-to-one closure.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
# The Network is the top-level container for every component, state, balance, and
# tracked value in this example.
SplitterNetwork = Network("Splitter Network")


# -----------------------------------------------------------------------------
# Chamber equilibrium lookup
# -----------------------------------------------------------------------------
# This Lookup wraps ThermoProp's Equilibrium object. The reactants are an H2/O2
# mixture at a fixed temperature and pressure. The returned equilibrium state is
# used as the separator inlet condition.
#
# The Lookup exposes states such as:
#
#     ChamberEquilibrium.composition
#     ChamberEquilibrium.pressure
#     ChamberEquilibrium.temperature
#     ChamberEquilibrium.density
#     ChamberEquilibrium.gas_constant
#
# These outputs behave like FullFlow state-like objects, so they can be connected
# directly into components and derived states.
ChamberEquilibrium = Lookup(
    "Combustion Chamber",
    SplitterNetwork,
    Equilibrium,
    reactants={"h2": 0.5, "o2": 0.5},
    mode="tp",
    temperature=2000,
    pressure=200 * 6894.76,
)

# Evaluate the lookup once before building composition dictionaries.
#
# The species list is intentionally not hardcoded. ThermoProp decides which
# equilibrium species are present, and this script uses that returned dictionary
# to create the separator composition equations.
#
# ChamberEquilibrium.composition is a lookup output. Its .value is the actual
# dictionary of equilibrium mass fractions:
#
#     {"H2O": 0.52, "O2": 0.46, ...}
#
# The rest of this example uses that plain dictionary.
chamber_composition = ChamberEquilibrium.composition.value


# -----------------------------------------------------------------------------
# Chamber source component
# -----------------------------------------------------------------------------
# The chamber is represented as a simple Component because this example is not
# solving combustion internally. ThermoProp has already provided the chamber gas
# state. The chamber acts as the upstream source for the separator network.
Chamber = Component(
    "Combustion Chamber",
    SplitterNetwork,
)

# The chamber composition is known, so keep it as a fixed dictionary of floats.
# It is not converted to State objects because the chamber composition is not an
# unknown in this FullFlow solve.
Chamber.composition = chamber_composition

# These properties are used by the inlet tube and by the simple density estimates
# for the separator node and exhaust stream.
Chamber.pressure = ChamberEquilibrium.pressure
Chamber.temperature = ChamberEquilibrium.temperature
Chamber.density = ChamberEquilibrium.density
Chamber.gas_constant = ChamberEquilibrium.gas_constant

# Initial guess for chamber outlet mass flow. The inlet FlowTube will solve this
# value from the pressure drop between the chamber and separator node.
Chamber.mass_flow_out = State(10.0)


# -----------------------------------------------------------------------------
# Separator node
# -----------------------------------------------------------------------------
# The separator node is a real lumped storage volume. In steady state, FullFlow
# drives its mass derivative to zero. Its pressure is the dynamic solve variable,
# and its density is estimated by ideal-gas pressure scaling:
#
#     rho_node = rho_chamber * P_node / P_chamber
separator_pressure = State(150 * 6894.76)
separator_density = Chamber.density * separator_pressure / Chamber.pressure

SeparatorNode = Volume(
    "Separator Node",
    SplitterNetwork,
    pressure=separator_pressure,
    volume=1.0,
    density=separator_density,
    mass_flow_in=Chamber.mass_flow_out,
)

# Keep the node temperature and gas constant equal to the chamber gas for this
# simple example.
SeparatorNode.temperature = Chamber.temperature
SeparatorNode.gas_constant = Chamber.gas_constant
SeparatorNode.density = separator_density


# -----------------------------------------------------------------------------
# Product stream
# -----------------------------------------------------------------------------
# The product stream is prescribed as pure water vapor. In a more realistic
# separator, this could be replaced by a non-ideal product composition, or the
# product composition could also be solved.
Product = Component(
    "Separated Product",
    SplitterNetwork,
)

# Missing species are treated as zero by the Composition component, so the pure
# H2O product does not need to explicitly list every chamber species.
Product.composition = {"H2O": 1.0}

# Product outlet pressure and thermodynamic assumptions. The gas constant is an
# approximate value for water vapor in J/kg-K.
Product.pressure = State(20 * 6894.76)
Product.temperature = Chamber.temperature
Product.gas_constant = State(461.52)

# Initial guess for product mass flow. The product outlet tube solves this value.
Product.mass_flow = State(1.0)

# Simple ideal-gas density estimates for the product tube. The upstream density
# uses separator pressure, while the downstream density uses product outlet
# pressure.
Product.upstream_density = SeparatorNode.pressure / (Product.gas_constant * Product.temperature)
Product.downstream_density = Product.pressure / (Product.gas_constant * Product.temperature)


# -----------------------------------------------------------------------------
# Exhaust stream
# -----------------------------------------------------------------------------
# The exhaust stream carries whatever gas remains after the product H2O stream is
# removed. Its mass flow and independent species fractions are solved.
Exhaust = Component(
    "Exhaust Products",
    SplitterNetwork,
)

# Convert the known chamber composition dictionary into a dictionary of States:
#
#     {"H2O": 0.52, "O2": 0.46, ...}
#
# becomes:
#
#     {"H2O": State(0.52), "O2": State(0.46), ...}
#
# These State values are initial guesses for the exhaust composition.
Exhaust.composition = State.from_iterable(chamber_composition)

# Exhaust outlet pressure, approximately atmospheric.
Exhaust.pressure = State(14.7 * 6894.76)

# Keep exhaust temperature and gas constant equal to the chamber mixture.
Exhaust.temperature = Chamber.temperature
Exhaust.gas_constant = Chamber.gas_constant

# Initial guess for exhaust mass flow. The exhaust outlet tube solves this value.
Exhaust.mass_flow = State(9.0)

# Simple ideal-gas density estimates for the exhaust tube.
Exhaust.upstream_density = SeparatorNode.pressure / (Exhaust.gas_constant * Exhaust.temperature)
Exhaust.downstream_density = Exhaust.pressure / (Exhaust.gas_constant * Exhaust.temperature)


# The separator node outlet flow is the sum of the two outlet streams. This closes
# the Volume mass balance.
SeparatorNode.mass_flow_out = Product.mass_flow + Exhaust.mass_flow


# -----------------------------------------------------------------------------
# Inlet flow tube
# -----------------------------------------------------------------------------
# This tube connects the chamber source to the separator node. The mass flow is
# solved from the pressure drop, density estimates, geometry, and friction factor.
InletTube = FlowTube(
    "Chamber to Separator Tube",
    SplitterNetwork,
    mass_flow=Chamber.mass_flow_out,
    upstream_static_pressure=Chamber.pressure,
    downstream_static_pressure=SeparatorNode.pressure,
    length=1.0,
    hydraulic_diameter=0.10,
    upstream_density=Chamber.density,
    downstream_density=SeparatorNode.density,
    friction_factor=0.02,
)


# -----------------------------------------------------------------------------
# Product outlet flow tube
# -----------------------------------------------------------------------------
# This tube connects the separator node to the pure H2O product outlet.
ProductTube = FlowTube(
    "Separated H2O Outlet Tube",
    SplitterNetwork,
    mass_flow=Product.mass_flow,
    upstream_static_pressure=SeparatorNode.pressure,
    downstream_static_pressure=Product.pressure,
    length=1.0,
    hydraulic_diameter=0.065,
    upstream_density=Product.upstream_density,
    downstream_density=Product.downstream_density,
    friction_factor=0.02,
)


# -----------------------------------------------------------------------------
# Exhaust outlet flow tube
# -----------------------------------------------------------------------------
# This tube connects the separator node to the exhaust outlet.
ExhaustTube = FlowTube(
    "Exhaust Outlet Tube",
    SplitterNetwork,
    mass_flow=Exhaust.mass_flow,
    upstream_static_pressure=SeparatorNode.pressure,
    downstream_static_pressure=Exhaust.pressure,
    length=1.0,
    hydraulic_diameter=0.18,
    upstream_density=Exhaust.upstream_density,
    downstream_density=Exhaust.downstream_density,
    friction_factor=0.02,
)


# -----------------------------------------------------------------------------
# Exhaust composition setup
# -----------------------------------------------------------------------------
# Mass fractions should sum to one. To avoid adding one redundant composition
# equation, H2O is removed from the independent solve dictionary and defined as
# the dependent remainder.
remainder = "H2O"

# These are the independent exhaust composition variables. The Composition
# component will create one residual for each key in this dictionary.
exhaust_solve = {
    species: fraction
    for species, fraction in Exhaust.composition.items()
    if species != remainder
}

# Replace the original H2O State with a derived State:
#
#     H2O = 1 - H - H2 - H2O2 - HO2 - O - O2 - O3 - OH
#
# H2O remains present in the full exhaust dictionary, but it is not directly
# iterated by the solver.
Exhaust.composition[remainder] = 1.0 - sum(exhaust_solve.values())


# -----------------------------------------------------------------------------
# Composition component
# -----------------------------------------------------------------------------
# Composition enforces steady-state conservation of carried mass fractions:
#
#     sum(mdot_in * x_in) - sum(mdot_out * x_out) = 0
#
# for each species listed in solve.
#
# Full dictionaries are passed to inlets and outlets. Only the non-H2O exhaust
# species are included in solve. Product.composition contains only H2O, so all
# missing non-H2O species are treated as zero in the product stream.
Separator = Composition(
    "H2O Separator",
    SplitterNetwork,
    inlets=[
        (Chamber.mass_flow_out, Chamber.composition),
    ],
    outlets=[
        (Product.mass_flow, Product.composition),
        (Exhaust.mass_flow, Exhaust.composition),
    ],
    solve=exhaust_solve,
)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------
# The steady-state solver determines separator pressure, all three tube mass
# flows, and the independent exhaust species fractions.
SteadyState(SplitterNetwork).solve(verbose=True)
