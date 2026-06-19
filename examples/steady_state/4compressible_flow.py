"""
Compressible Diffuser, Volume, and Orifice Network
==================================================

This example demonstrates a simple compressible-flow network using FullFlow
and ThermoProp.

Network Layout
--------------
    GN2 Source
        |
        v
    Isentropic Diffuser
        |
        v
    Lumped Volume / Node
        |
        v
    Compressible Orifice / Nozzle
        |
        v
    Atmosphere

The network models gaseous nitrogen flowing from a source reservoir through an
isentropic diffuser into an internal node volume, and then out through a
compressible orifice to atmosphere.

The diffuser determines the inlet and outlet Mach numbers and the mass flow
rate required to connect the source static state to the node static pressure.

The volume enforces conservation of mass and energy at the intermediate node.
It receives mass flow and total enthalpy from the diffuser and passes mass flow
and total enthalpy to the downstream orifice.

The downstream orifice uses compressible choked/non-choked flow relations
based on upstream total pressure and total temperature.

A balance is added to vary the diffuser inlet area until the diffuser inlet Mach
number reaches a target value of 0.7.

Together, the components form a coupled nonlinear system:

    Isentropic Diffuser:
        Uses upstream static pressure/temperature, downstream static pressure,
        area ratio, gamma, and gas constant to solve for Mach numbers and mass
        flow.

    Volume:
        mass_flow_in - mass_flow_out = 0

        total_enthalpy_in - total_enthalpy_out = 0

    Compressible Orifice:
        Uses upstream total pressure, upstream total temperature, downstream
        pressure, discharge coefficient, area, gamma, and gas constant to solve
        the outlet mass flow.

    Balance:
        Diff.upstream_mach_number = 0.7

The steady-state solver iterates on the coupled flow variables, volume state,
and diffuser inlet area until the diffuser, volume, orifice, and balance
equations are all satisfied simultaneously.

Demonstrates
------------
- ThermoProp ideal-gas property lookups using Lookup
- Compressible flow through an isentropic diffuser
- Lumped-volume mass and energy conservation
- Compressible choked/non-choked orifice flow
- Derived total pressure and total temperature states
- Algebraic balance on a geometric design variable
- Automatic steady-state network solving
"""

import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Create network
#
# The Network object owns all states, components, balances, and property lookups
# used in this example.
# -----------------------------------------------------------------------------

CompressibleNetwork = Network("Compressible Network")


# -----------------------------------------------------------------------------
# Source fluid property lookup
#
# This defines the upstream GN2 source condition.
#
# ThermoProp's IdealGas model provides:
#
#     SourceFluid.pressure
#     SourceFluid.temperature
#     SourceFluid.enthalpy
#     SourceFluid.gamma
#     SourceFluid.gas_constant
#
# These outputs behave like FullFlow states and can be connected directly into
# components.
#
# This source is treated as a fixed boundary condition:
#
#     P = 3e5 Pa
#     T = 300 K
# -----------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source Fluid",
    CompressibleNetwork,
    IdealGas,
    "gn2",
    pressure=3e5, 
    temperature=300,
)


# -----------------------------------------------------------------------------
# Node fluid property lookup
#
# This lookup represents the thermodynamic state inside the intermediate node.
#
# The node pressure is given an initial guess of 5e5 Pa. Because this pressure is
# also passed into the Volume component, it can become part of the coupled
# solution depending on how the Volume component registers its iteration states.
#
# The priority setting tells the Lookup which thermodynamic input pair should be
# preferred when multiple possible inputs exist.
#
# Here:
#
#     priority=("enthalpy", "temperature")
#
# means the lookup should prefer enthalpy as the active thermodynamic input when
# enthalpy is available, while temperature can still serve as an initial/fallback
# input.
#
# This is useful because the Volume solves an energy balance in terms of
# enthalpy, and the temperature should then be updated consistently from the
# thermodynamic property model.
# -----------------------------------------------------------------------------

NodeFluid = Lookup(
    "Node Fluid",
    CompressibleNetwork,
    IdealGas,
    "gn2",
    pressure=5e5, 
    temperature=300,
    priority=("enthalpy", "temperature")
)


# -----------------------------------------------------------------------------
# Isentropic diffuser
#
# The diffuser connects the source static state to the node static pressure.
#
# Inputs:
#     upstream_static_pressure       Source static pressure [Pa]
#     upstream_static_temperature    Source static temperature [K]
#     downstream_static_pressure     Node static pressure [Pa]
#     inlet_cross_sectional_area     Diffuser inlet area [m²]
#     outlet_cross_sectional_area    Diffuser outlet area [m²]
#     specific_heat_ratio            gamma [-]
#     gas_constant                   gas constant [J/kg-K]
#     upstream_static_enthalpy       source static enthalpy [J/kg]
#
# Outputs created by the component include:
#     Diff.mass_flow
#     Diff.total_enthalpy
#     Diff.upstream_mach_number
#     Diff.downstream_mach_number
#
# The diffuser is isentropic, so the total enthalpy is conserved between the
# inlet and outlet. The component computes the flow state that satisfies the
# pressure change and area change.
# -----------------------------------------------------------------------------

Diff = IsentropicDiffuser(
    "Diff",
    CompressibleNetwork,
    upstream_static_pressure=SourceFluid.pressure,
    upstream_static_temperature=SourceFluid.temperature,
    downstream_static_pressure=NodeFluid.pressure,
    inlet_cross_sectional_area=(np.pi/4) * (0.75**2),
    outlet_cross_sectional_area=(np.pi/4) * (1**2),
    specific_heat_ratio=SourceFluid.gamma,
    gas_constant=SourceFluid.gas_constant,
    upstream_static_enthalpy=SourceFluid.enthalpy
)


# -----------------------------------------------------------------------------
# Lumped volume / internal node
#
# The Volume component enforces conservation at the intermediate node.
#
# It receives:
#
#     mass_flow_in        = Diff.mass_flow
#     total_enthalpy_in   = Diff.total_enthalpy
#
# It solves/provides:
#
#     Node.mass_flow_out
#     Node.total_enthalpy_out
#
# These outlet quantities are then passed to the downstream compressible orifice.
#
# The Volume is what couples the upstream diffuser and downstream orifice:
#
#     Diff.mass_flow = Node.mass_flow_out
#
# and, when energy is active:
#
#     Diff.total_enthalpy = Node.total_enthalpy_out
#
# The node pressure, enthalpy, and temperature are connected to the NodeFluid
# property lookup so that the thermodynamic state remains consistent with the
# conservation equations.
# -----------------------------------------------------------------------------

Node = Volume(
    "Node",
    CompressibleNetwork,
    pressure=NodeFluid.pressure,
    enthalpy=NodeFluid.enthalpy,
    temperature=NodeFluid.temperature,
    mass_flow_in=Diff.mass_flow,
    total_enthalpy_in=Diff.total_enthalpy,
    volume=1,
)


# -----------------------------------------------------------------------------
# Convert node static state to total state for the downstream orifice
#
# The compressible orifice expects upstream total pressure and total temperature.
#
# The node lookup gives static pressure and static temperature. To construct the
# total state, the downstream Mach number from the diffuser is used:
#
#     Po = P (1 + (gamma - 1) M² / 2)^(gamma / (gamma - 1))
#
#     To = T (1 + (gamma - 1) M² / 2)
#
# These are isentropic perfect-gas relations.
#
# M is taken from the diffuser outlet because the diffuser outlet feeds directly
# into the node/orifice inlet.
# -----------------------------------------------------------------------------

M = Diff.downstream_mach_number
g = NodeFluid.gamma

po = NodeFluid.pressure * (1.0 + 0.5 * (g - 1.0) * M**2) ** (g / (g - 1.0))
To = NodeFluid.temperature * (1 + (g-1)/2 * M**2)


# -----------------------------------------------------------------------------
# Compressible orifice / nozzle
#
# This component models the discharge from the node to atmosphere.
#
# Inputs:
#     upstream_total_pressure        Total pressure at the node/orifice inlet [Pa]
#     upstream_total_temperature     Total temperature at the node/orifice inlet [K]
#     downstream_pressure            Back pressure / atmosphere [Pa]
#     discharge_coefficient          Cd [-]
#     cross_sectional_area           Flow area [m²]
#     gas_constant                   gas constant [J/kg-K]
#     specific_heat_ratio            gamma [-]
#
# Optional energy coupling:
#     upstream_static_enthalpy       Node static enthalpy [J/kg]
#     upstream_static_temperature    Node static temperature [K]
#     total_enthalpy                 Connected to Node.total_enthalpy_out
#
# Coupled mass flow:
#     mass_flow                      Connected to Node.mass_flow_out
#
# Since Nozzle.mass_flow is the same state as Node.mass_flow_out, the Volume and
# CompressibleOrifice solve the outlet flow together.
#
# The orifice internally determines whether the flow is choked based on the
# pressure ratio:
#
#     P_downstream / Po
#
# and uses the appropriate compressible mass-flow relation.
# -----------------------------------------------------------------------------

Nozzle = CompressibleOrifice(
    "Nozzle",
    CompressibleNetwork,
    upstream_total_pressure=po,
    upstream_total_temperature=To,
    downstream_pressure=101325,
    discharge_coefficient=1,
    cross_sectional_area=(np.pi/4) * (0.75**2),
    gas_constant=NodeFluid.gas_constant,
    specific_heat_ratio=NodeFluid.gamma,
    upstream_static_enthalpy=NodeFluid.enthalpy,
    upstream_static_temperature=NodeFluid.temperature,
    total_enthalpy=Node.total_enthalpy_out,
    mass_flow=Node.mass_flow_out
)


# -----------------------------------------------------------------------------
# Diffuser inlet-area balance
#
# This Balance varies the diffuser inlet area until the diffuser inlet Mach number
# reaches the target value of 0.7.
#
# Unknown:
#     Diff.inlet_cross_sectional_area
#
# Residual:
#     Diff.upstream_mach_number - 0.7 = 0
#
# In other words, the solver adjusts the inlet area until:
#
#     Diff.upstream_mach_number = 0.7
#
# This is a design-style constraint. Instead of specifying the inlet area and
# accepting whatever Mach number results, the Mach number is specified and the
# required inlet area is solved.
# -----------------------------------------------------------------------------

InletBalance = Balance(
    "Inlet Area Balance",
    CompressibleNetwork,
    variable=Diff.inlet_cross_sectional_area,
    function=Diff.upstream_mach_number - 0.7
)


# -----------------------------------------------------------------------------
# Solve network
#
# The steady-state solver evaluates all property lookups, components, and
# balances, then iterates on the unknown states until all residuals are driven
# below the requested tolerances.
#
# verbose=True prints the solver summary, residuals, and solution variables.
# -----------------------------------------------------------------------------

SteadyState(CompressibleNetwork).solve(verbose=True)