"""
Transient Solid Heat Transfer With Natural Convection
====================================================

This example shows a minimal transient heat-transfer model using only built-in
FullFlow components and a ThermoProp material lookup.

Physical layout
---------------

    Fixed-temperature side                         Ambient air
        T_hot                                      T_air
          │                                          │
          │ q_hot = h_hot A_hot (T_hot - T_metal)    │
          ▼                                          │
    ┌────────────────┐                              │
    │                │                              │
    │ Aluminum solid │──── q_air = h_air A_air (T_air - T_metal)
    │  T_metal(t)   │                              │
    │                │                              ▼
    └────────────────┘

Model notes
-----------

The metal is treated as one lumped thermal mass:

    dT_metal/dt = Q_net / (m Cp)

where:

    Q_net = q_hot + q_air

The fixed-temperature side uses a prescribed heat-transfer coefficient.
The air side uses the built-in NaturalConvection component to calculate h_air
from the current wall-to-air temperature difference.

Sign convention
---------------

The built-in Convection component uses:

    q = h A (T_fluid - T_surface)

Therefore:

    positive q  -> heat enters the solid
    negative q  -> heat leaves the solid

This example intentionally uses no custom components. The two surface heat
rates are calculated by built-in Convection components, and the net heat rate is
just a derived State made by adding those two heat-rate states together.
"""


import h5py

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
HeatTransferNetwork = Network("Solid Heat Transfer Transient")

filename = "11heat_transfer"


# ---------------------------------------------------------------------------
# Solid node state and geometry
# ---------------------------------------------------------------------------
# This is the only transient state in the model. The Solid component integrates
# this temperature from the net heat rate and the metal heat capacity.
metal_temperature = State(320.0)      # K
metal_mass = 1.0                      # kg

# Simple flat-plate geometry. The characteristic length is used by the natural
# convection correlation and by Solid for a simple Biot-number calculation.
hot_side_area = 0.020                 # m^2
air_side_area = 0.020                 # m^2
characteristic_length = 0.050         # m


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
# One side of the solid sees a known temperature with a prescribed heat-transfer
# coefficient. This is a boundary condition, not another dynamic node.
hot_side_temperature = State(500.0)   # K
hot_side_heat_transfer_coefficient = State(80.0)  # W/m^2/K

# The other side loses heat to still ambient air by natural convection. The air
# properties are held constant to keep this example focused on transient heat
# storage in the metal.
air_temperature = State(300.0)        # K
air_density = State(1.177)            # kg/m^3
air_specific_heat = State(1007.0)     # J/kg/K
air_dynamic_viscosity = State(1.85e-5)  # Pa*s
air_conductivity = State(0.0263)      # W/m/K
air_thermal_expansion_coefficient = State(1.0 / air_temperature.value)  # 1/K

biot_number = State(0.0)


# ---------------------------------------------------------------------------
# ThermoProp material lookup
# ---------------------------------------------------------------------------
# The Material lookup follows metal_temperature during the transient. The Solid
# node uses Aluminum.specific_heat as Cp, so Cp updates as the metal heats up.
Aluminum = Lookup(
    "Aluminum 6061 Properties",
    HeatTransferNetwork,
    Material,
    "Aluminum 6061",
    temperature=metal_temperature,
)


# ---------------------------------------------------------------------------
# Built-in heat-transfer components
# ---------------------------------------------------------------------------
# Heat from the fixed-temperature side into the metal.
HotSideConvection = Convection(
    "Hot Side Convection",
    HeatTransferNetwork,
    surface_temperature=metal_temperature,
    fluid_temperature=hot_side_temperature,
    convective_area=hot_side_area,
    convection_coefficient=hot_side_heat_transfer_coefficient,
)

# Natural-convection coefficient on the air side.
AirSideNaturalConvection = NaturalConvection(
    "Air Side Natural Convection",
    HeatTransferNetwork,
    wall_temperature=metal_temperature,
    fluid_temperature=air_temperature,
    characteristic_length=characteristic_length,
    fluid_density=air_density,
    fluid_specific_heat=air_specific_heat,
    fluid_dynamic_viscosity=air_dynamic_viscosity,
    fluid_conductivity=air_conductivity,
    thermal_expansion_coefficient=air_thermal_expansion_coefficient,
)

# Heat from ambient air into the metal. Since the air is colder than the metal
# for most of this run, this heat rate will usually be negative.
AirSideConvection = Convection(
    "Air Side Convection",
    HeatTransferNetwork,
    surface_temperature=metal_temperature,
    fluid_temperature=air_temperature,
    convective_area=air_side_area,
    convection_coefficient=AirSideNaturalConvection.convection_coefficient,
)

# No custom heat-rate summing component is needed. FullFlow States can form
# derived states directly.
net_heat_rate = HotSideConvection.heat_rate + AirSideConvection.heat_rate


# ---------------------------------------------------------------------------
# Dynamic solid node
# ---------------------------------------------------------------------------
# Solid owns the transient equation:
#
#     dT_metal/dt = net_heat_rate / (metal_mass * Aluminum.specific_heat)
#
# The conductivity and natural-convection coefficient are optional inputs here;
# they are only included so the Solid node also reports a Biot number.
MetalNode = Solid(
    "Aluminum Solid Node",
    HeatTransferNetwork,
    temperature=metal_temperature,
    mass=metal_mass,
    specific_heat=Aluminum.specific_heat,
    characteristic_length=characteristic_length,
    thermal_conductivity=Aluminum.thermal_conductivity,
    convection_coefficient=AirSideNaturalConvection.convection_coefficient,
    biot_number=biot_number,
    heat_rate=net_heat_rate,
)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
HeatTransferNetwork.track("Metal Temperature K", metal_temperature)
HeatTransferNetwork.track("Hot Side Temperature K", hot_side_temperature)
HeatTransferNetwork.track("Air Temperature K", air_temperature)
HeatTransferNetwork.track("Aluminum Cp J per kg K", Aluminum.specific_heat)
HeatTransferNetwork.track("Aluminum Conductivity W per m K", Aluminum.thermal_conductivity)
HeatTransferNetwork.track("Hot Side Heat Rate W", HotSideConvection.heat_rate)
HeatTransferNetwork.track("Air Side Heat Rate W", AirSideConvection.heat_rate)
HeatTransferNetwork.track("Net Heat Rate W", net_heat_rate)
HeatTransferNetwork.track("Natural Convection Coefficient W per m2 K", AirSideNaturalConvection.convection_coefficient)
HeatTransferNetwork.track("Rayleigh Number", AirSideNaturalConvection.rayleigh_number)
HeatTransferNetwork.track("Biot Number", biot_number)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------
Transient(HeatTransferNetwork).solve(
    dt=1.0,
    t_final=900.0,
    filename=filename,
    save_dt=5.0,
    verbose=True,
)