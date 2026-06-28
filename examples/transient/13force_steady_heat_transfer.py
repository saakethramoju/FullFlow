"""
Simple force_steady transient example.

This example shows how FullFlow's transient force_steady option works.

Physical model
--------------

A single metal lump is connected to two fixed-temperature environments:

    Hot boundary ---- convection ----+
                                     |
                                  Metal lump
                                     |
    Cold boundary --- convection ----+

The metal has real thermal storage:

    dT_metal/dt = heat_rate / (mass * Cp)

The hot boundary temperature ramps upward during the transient.

Two separate networks are solved:

1. Dynamic metal
   The Solid is integrated normally, so the metal temperature lags.

2. Force-steady metal
   The Solid component is listed in force_steady, so the transient solver
   replaces the Solid's transient corrector equation with:

       temperature_dot = 0

   Since temperature_dot = heat_rate / (mass * Cp), this is equivalent to:

       heat_rate = 0

   The metal becomes a massless thermal junction and instantly moves to the
   quasi-steady temperature where heat in from the hot side equals heat out to
   the cold side.
"""

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Shared settings
# -----------------------------------------------------------------------------

hot_temperature_initial = 300.0
hot_temperature_final = 500.0

cold_temperature = 300.0

hot_convection_coefficient = 20.0
cold_convection_coefficient = 10.0

hot_area = 1.0
cold_area = 1.0

metal_mass = 5.0


def hot_temperature_schedule(t):
    """Step the hot boundary temperature after the initialized condition."""
    if t < 1.0:
        return hot_temperature_initial

    return hot_temperature_final


def build_network(name):
    """Build one copy of the simple thermal network."""

    ThermalNetwork = Network(name)

    HotTemperature = State(hot_temperature_initial)
    ColdTemperature = State(cold_temperature)

    HotTemperatureSchedule = Sequence(
        "Hot Temperature Schedule",
        ThermalNetwork,
        target=HotTemperature,
        function=hot_temperature_schedule,
    )

    MetalMaterial = Lookup(
        "Metal Material",
        ThermalNetwork,
        Material,
        "c101",
        temperature=hot_temperature_initial,
    )

    MetalNode = Solid(
        "Metal Node",
        ThermalNetwork,
        temperature=MetalMaterial.temperature,
        mass=metal_mass,
        specific_heat=MetalMaterial.specific_heat,
    )

    HotSideConvection = Convection(
        "Hot Side Convection",
        ThermalNetwork,
        surface_temperature=MetalNode.temperature,
        fluid_temperature=HotTemperature,
        convection_coefficient=hot_convection_coefficient,
        convective_area=hot_area,
    )

    ColdSideConvection = Convection(
        "Cold Side Convection",
        ThermalNetwork,
        surface_temperature=MetalNode.temperature,
        fluid_temperature=ColdTemperature,
        convection_coefficient=cold_convection_coefficient,
        convective_area=cold_area,
    )

    MetalNode.heat_rate = (
        HotSideConvection.heat_rate
        + ColdSideConvection.heat_rate
    )

    # This is the temperature the metal would have if heat_rate = 0.
    quasi_steady_temperature = (
        hot_convection_coefficient * hot_area * HotTemperature
        + cold_convection_coefficient * cold_area * ColdTemperature
    ) / (
        hot_convection_coefficient * hot_area
        + cold_convection_coefficient * cold_area
    )

    ThermalNetwork.track("Hot Boundary Temperature [K]", HotTemperature)
    ThermalNetwork.track("Cold Boundary Temperature [K]", ColdTemperature)
    ThermalNetwork.track("Metal Temperature [K]", MetalNode.temperature)
    ThermalNetwork.track("Quasi-Steady Metal Temperature [K]", quasi_steady_temperature)
    ThermalNetwork.track("Metal Heat Rate [W]", MetalNode.heat_rate)
    ThermalNetwork.track("Hot Side Heat Rate [W]", HotSideConvection.heat_rate)
    ThermalNetwork.track("Cold Side Heat Rate [W]", ColdSideConvection.heat_rate)

    return ThermalNetwork, MetalNode


# -----------------------------------------------------------------------------
# Case 1: normal dynamic solid
# -----------------------------------------------------------------------------

DynamicNetwork, DynamicMetalNode = build_network("Dynamic Solid Example")

SteadyState(DynamicNetwork).solve(
    verbose=True,
    filename="dynamic_solid_example",
)

Transient(DynamicNetwork).solve(
    dt=0.1,
    t_final=20.0,
    save_dt=0.1,
    filename="dynamic_solid_example",
    verbose=True,
)


# -----------------------------------------------------------------------------
# Case 2: force-steady solid
# -----------------------------------------------------------------------------

ForceSteadyNetwork, ForceSteadyMetalNode = build_network("Force Steady Solid Example")

SteadyState(ForceSteadyNetwork).solve(
    verbose=True,
    filename="force_steady_solid_example",
)

Transient(ForceSteadyNetwork).solve(
    dt=0.1,
    t_final=20.0,
    save_dt=0.1,
    filename="force_steady_solid_example",
    verbose=True,
    force_steady=[ForceSteadyMetalNode],
)