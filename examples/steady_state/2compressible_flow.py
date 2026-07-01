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
# -----------------------------------------------------------------------------

CompressibleNetwork = Network("Compressible Network")


# -----------------------------------------------------------------------------
# Source fluid property lookup
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
# -----------------------------------------------------------------------------

NodeFluid = Lookup(
    "Node Fluid",
    CompressibleNetwork,
    IdealGas,
    "gn2",
    pressure=5e5,
    temperature=300,
    priority=("enthalpy", "temperature"),
)


# -----------------------------------------------------------------------------
# Geometry
#
# FullFlow uses SI units internally. These diameters are specified in inches
# and converted to meters before calculating areas.
#
# The diffuser inlet area is still allowed to vary because of the Balance below.
# These values are only the initial geometry guesses / fixed downstream geometry.
# -----------------------------------------------------------------------------

IN_TO_M = 0.0254

D_diff_in = 0.75 * IN_TO_M
D_diff_out = 1.00 * IN_TO_M
D_nozzle = 0.75 * IN_TO_M

A_diff_in = np.pi * D_diff_in**2 / 4.0
A_diff_out = np.pi * D_diff_out**2 / 4.0
A_nozzle = np.pi * D_nozzle**2 / 4.0


# -----------------------------------------------------------------------------
# Isentropic diffuser
# -----------------------------------------------------------------------------

Diff = IsentropicDiffuser(
    "Diff",
    CompressibleNetwork,
    upstream_static_pressure=SourceFluid.pressure,
    upstream_static_temperature=SourceFluid.temperature,
    downstream_static_pressure=NodeFluid.pressure,
    inlet_cross_sectional_area=A_diff_in,
    outlet_cross_sectional_area=A_diff_out,
    specific_heat_ratio=SourceFluid.gamma,
    gas_constant=SourceFluid.gas_constant,
    upstream_static_enthalpy=SourceFluid.enthalpy,
)


# -----------------------------------------------------------------------------
# Lumped storage volume / internal node
#
# SteadyState drives Node.mass_dot and Node.total_internal_energy_dot to zero.
# Transient would integrate the same mass and energy storage equations.
# -----------------------------------------------------------------------------

Node = Volume(
    "Node",
    CompressibleNetwork,
    pressure=NodeFluid.pressure,
    volume=1,
    density=NodeFluid.density,
    mass_flow_in=Diff.mass_flow,
    enthalpy=NodeFluid.enthalpy,
    temperature=NodeFluid.temperature,
    internal_energy=NodeFluid.internal_energy,
    total_enthalpy_in=Diff.total_enthalpy,
)


# -----------------------------------------------------------------------------
# Convert node static state to total state for the downstream orifice
# -----------------------------------------------------------------------------

M = Diff.downstream_mach_number
g = NodeFluid.gamma

po = NodeFluid.pressure * (1.0 + 0.5 * (g - 1.0) * M**2) ** (g / (g - 1.0))
To = NodeFluid.temperature * (1.0 + 0.5 * (g - 1.0) * M**2)


# -----------------------------------------------------------------------------
# Compressible orifice / nozzle
# -----------------------------------------------------------------------------

Nozzle = CompressibleOrifice(
    "Nozzle",
    CompressibleNetwork,
    upstream_total_pressure=po,
    upstream_total_temperature=To,
    downstream_pressure=101325,
    discharge_coefficient=1,
    cross_sectional_area=A_nozzle,
    gas_constant=NodeFluid.gas_constant,
    specific_heat_ratio=NodeFluid.gamma,
    upstream_static_enthalpy=NodeFluid.enthalpy,
    upstream_static_temperature=NodeFluid.temperature,
    total_enthalpy=Node.total_enthalpy_out,
    mass_flow=Node.mass_flow_out,
)


# -----------------------------------------------------------------------------
# Diffuser inlet-area balance
#
# This is intentionally kept. The solver will vary Diff.inlet_cross_sectional_area
# until Diff.upstream_mach_number reaches 0.7.
# -----------------------------------------------------------------------------

InletBalance = Balance(
    "Inlet Area Balance",
    CompressibleNetwork,
    variable=Diff.inlet_cross_sectional_area,
    function=Diff.upstream_mach_number - 0.7,
)


# -----------------------------------------------------------------------------
# Solve network
# -----------------------------------------------------------------------------

SteadyState(CompressibleNetwork).solve(verbose=True)