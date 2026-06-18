"""
Steady-state two-volume GN2 vent network example.

This example models a pressurized GN2 source feeding two connected flow sections
and venting through a compressible orifice.

Network layout
--------------
    SourceFluid
        |
        v
    Tube1
        |
        v
    GasVolume1 / Volume1Fluid
        |
        v
    Tube2
        |
        v
    GasVolume2 / Volume2Fluid
        |
        v
    VentOrifice
        |
        v
    Ambient

Important modeling notes
------------------------
- `Volume1Fluid` and `Volume2Fluid` are thermodynamic property lookups for the
  gas state at each volume/control node.

- `GasVolume1` and `GasVolume2` enforce steady-state mass and energy balance.

- `GasVolume2` is not meant to represent a large stagnant tank. It represents a
  tube-exit control station just upstream of the vent orifice. Because the flow
  is still moving there, the vent orifice is fed using `Tube2` downstream total
  pressure and total temperature.

- The orifice receives the upstream static enthalpy and temperature separately
  so it can compute total enthalpy on the same ThermoProp reference basis as the
  rest of the network.

- `priority=("enthalpy", "temperature")` lets the property lookup initialize
  from pressure-temperature, then switch to pressure-enthalpy once the volume
  energy equation starts solving enthalpy.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

VentSystem = Network("Vent Network")


# -----------------------------------------------------------------------------
# Source gas state
#
# This is the high-pressure GN2 source feeding the first tube.
# -----------------------------------------------------------------------------

SourceFluid = Lookup(
    "Source Fluid",
    VentSystem,
    IdealGas,
    fluid="gn2",
    pressure=4.2e7,
    temperature=500,
)


# -----------------------------------------------------------------------------
# Volume 1 gas state
#
# The initial pressure-temperature pair gives the lookup a valid starting state.
# During solving, the volume energy equation drives `Volume1Fluid.enthalpy`, so
# the lookup switches to pressure-enthalpy.
# -----------------------------------------------------------------------------

Volume1Fluid = Lookup(
    "Volume 1 Fluid",
    VentSystem,
    IdealGas,
    fluid=SourceFluid.fluid,
    pressure=3e7,
    temperature=290,
    priority=("enthalpy", "temperature"),
)


# -----------------------------------------------------------------------------
# Volume 2 / vent-inlet gas state
#
# This is the thermodynamic state at the second control station, just upstream of
# the vent orifice.
# -----------------------------------------------------------------------------

Volume2Fluid = Lookup(
    "Vent 2 Fluid",
    VentSystem,
    IdealGas,
    fluid="gn2",
    pressure=3e7,
    temperature=300,
    priority=("enthalpy", "temperature"),
)


# -----------------------------------------------------------------------------
# Shared tube geometry
# -----------------------------------------------------------------------------

tube_diameter = 1 / 39.3701
tube_area = (np.pi / 4) * tube_diameter**2


# -----------------------------------------------------------------------------
# Tube 1: source -> volume 1
#
# This tube solves its mass flow from the momentum balance between the source
# state and Volume 1 state.
# -----------------------------------------------------------------------------

Tube1 = CompressibleFlowTube(
    "Pressurant Tube 1",
    VentSystem,
    mass_flow=30,
    upstream_static_pressure=SourceFluid.pressure,
    upstream_static_temperature=SourceFluid.temperature,
    upstream_density=SourceFluid.density,
    downstream_static_pressure=Volume1Fluid.pressure,
    downstream_static_temperature=Volume1Fluid.temperature,
    downstream_density=Volume1Fluid.density,
    downstream_speed_of_sound=Volume1Fluid.speed_of_sound,
    length=10,
    inner_diameter=tube_diameter,
    friction_factor=0.002,
    upstream_static_enthalpy=SourceFluid.enthalpy,
)


# -----------------------------------------------------------------------------
# Tube 1 friction factor
#
# Churchill computes the Darcy friction factor for Tube 1. The friction factor is
# tied directly to `Tube1.friction_factor`, so the tube momentum residual and the
# friction correlation solve together.
# -----------------------------------------------------------------------------

Tube1Friction = Churchill(
    "Tube 1 Churchill Friction Factor",
    VentSystem,
    mass_flow=Tube1.mass_flow,
    friction_factor=Tube1.friction_factor,
    hydraulic_diameter=Tube1.inner_diameter,
    cross_sectional_area=tube_area,
    dynamic_viscosity=SourceFluid.dynamic_viscosity,
    roughness=1e-7,
)


# -----------------------------------------------------------------------------
# Volume 1
#
# The volume enforces:
#
#   mass_flow_in - mass_flow_out = 0
#
# and, because enthalpy and total enthalpy are connected:
#
#   mdot_in * h0_in - mdot_out * h0_out = 0
#
# `mass_flow_out=30` is only the initial guess. It becomes the shared state used
# by Tube 2.
# -----------------------------------------------------------------------------

GasVolume1 = Volume(
    "Gas Volume 1",
    VentSystem,
    pressure=Volume1Fluid.pressure,
    enthalpy=Volume1Fluid.enthalpy,
    temperature=Volume1Fluid.temperature,
    volume=1,
    mass_flow_in=Tube1.mass_flow,
    mass_flow_out=30,
    total_enthalpy_in=Tube1.total_enthalpy,
)


# -----------------------------------------------------------------------------
# Tube 2: volume 1 -> volume 2 / orifice inlet station
#
# Tube 2 uses the Volume 1 state as its upstream state and the Volume 2 state as
# its downstream state.
#
# `total_enthalpy=GasVolume1.total_enthalpy_out` connects the branch total
# enthalpy to Volume 1's outlet energy balance.
#
# `specific_heat_ratio` and `downstream_speed_of_sound` allow Tube 2 to compute
# downstream total pressure and total temperature, which are then passed to the
# vent orifice.
# -----------------------------------------------------------------------------

Tube2 = CompressibleFlowTube(
    "Pressurant Tube 2",
    VentSystem,
    mass_flow=GasVolume1.mass_flow_out,
    upstream_static_pressure=GasVolume1.pressure,
    upstream_static_temperature=GasVolume1.temperature,
    upstream_density=Volume1Fluid.density,
    downstream_static_pressure=Volume2Fluid.pressure,
    downstream_static_temperature=Volume2Fluid.temperature,
    downstream_density=Volume2Fluid.density,
    downstream_speed_of_sound=Volume2Fluid.speed_of_sound,
    specific_heat_ratio=Volume2Fluid.specific_heat_ratio,
    length=6,
    inner_diameter=tube_diameter,
    friction_factor=Tube1.friction_factor,
    upstream_static_enthalpy=Volume1Fluid.enthalpy,
    total_enthalpy=GasVolume1.total_enthalpy_out,
)


# -----------------------------------------------------------------------------
# Tube 2 friction factor
#
# This uses the Volume 1 gas viscosity as the representative viscosity for the
# second tube.
# -----------------------------------------------------------------------------

Tube2Friction = Churchill(
    "Tube 2 Churchill Friction Factor",
    VentSystem,
    mass_flow=Tube2.mass_flow,
    friction_factor=Tube2.friction_factor,
    hydraulic_diameter=Tube2.inner_diameter,
    cross_sectional_area=tube_area,
    dynamic_viscosity=Volume1Fluid.dynamic_viscosity,
    roughness=1e-7,
)


# -----------------------------------------------------------------------------
# Volume 2 / orifice inlet control station
#
# This is a compact control node immediately before the orifice. It enforces mass
# and energy balance between Tube 2 and the vent orifice.
# -----------------------------------------------------------------------------

GasVolume2 = Volume(
    "Gas Volume 2",
    VentSystem,
    pressure=Volume2Fluid.pressure,
    enthalpy=Volume2Fluid.enthalpy,
    temperature=Volume2Fluid.temperature,
    volume=1,
    mass_flow_in=Tube2.mass_flow,
    total_enthalpy_in=Tube2.total_enthalpy,
)


# -----------------------------------------------------------------------------
# Vent orifice
#
# The orifice is fed by Tube 2 downstream total conditions because this node is
# representing a moving tube-exit station, not a stagnant plenum.
#
# The static enthalpy and static temperature are also passed in so the orifice
# can compute total enthalpy using:
#
#   h0 = h_static + cp * (T0 - T_static)
#
# instead of using raw cp*T0, which would use a different enthalpy reference than
# ThermoProp.
# -----------------------------------------------------------------------------

VentOrifice = IsentropicCompressibleOrifice(
    "Vent Orifice",
    VentSystem,
    upstream_total_pressure=Tube2.downstream_total_pressure,
    upstream_total_temperature=Tube2.downstream_total_temperature,
    upstream_static_enthalpy=GasVolume2.enthalpy,
    upstream_static_temperature=GasVolume2.temperature,
    downstream_pressure=101325,
    discharge_coefficient=1,
    cross_sectional_area=0.5 * Tube2Friction.cross_sectional_area,
    specific_gas_constant=Volume2Fluid.gas_constant,
    specific_heat_ratio=Volume2Fluid.specific_heat_ratio,
    mass_flow=GasVolume2.mass_flow_out,
    total_enthalpy=GasVolume2.total_enthalpy_out,
)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------

SteadyState(VentSystem).solve(
    verbose=True,
    )