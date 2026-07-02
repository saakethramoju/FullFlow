"""
Transient Cavitating-Venturi Feed-System Example
===============================================

This example extends the steady-state cavitating-venturi injector model into a
real transient feed-system model.

The main differences from the steady-state version are:

1. The liquid nodes are finite Volume components, so their pressures evolve from
   mass storage instead of being purely algebraic junction pressures.
2. The upstream feed lines are Darcy-Weisbach components with finite length, so
   their mass flows have inertia and cannot change instantaneously.
3. The injector restrictions are given a small effective length, so injector
   flow also has a finite response time.
4. Chamber gas properties come from a precomputed RP-1/LOX combustion-products
   map instead of directly calling equilibrium chemistry during every solver
   iteration.
5. The injector Cd values open smoothly at different rates. The initial Cd
   values are low enough that the steady-state initialization starts in a
   noncavitating state. As injector demand rises, the venturi downstream
   pressures fall and cavitation begins when a downstream pressure falls below
   its critical downstream pressure.

Physical Layout
---------------

Fuel side:

    RP-1 Source
        |
        v
    Fuel Feed Line
        - Darcy-Weisbach loss
        - line inertia
        |
        v
    Fuel Venturi Upstream Plenum
        - transient liquid storage volume
        |
        v
    Fuel Cavitating Venturi
        |
        v
    Fuel Injector Manifold
        - transient liquid storage volume
        |
        v
    Fuel Injector Orifices
        - smooth Cd opening
        - small effective inertial length
        |
        v

                         Combustion Chamber  --->  Isentropic Nozzle
        ^
        |
    Ox Injector Orifices
        ^
        |
    Ox Injector Manifold
        ^
        |
    Ox Cavitating Venturi
        ^
        |
    Ox Venturi Upstream Plenum
        ^
        |
    Ox Feed Line
        ^
        |
    LOX Source

Cavitation indicator
--------------------

For each venturi, FullFlow computes:

    critical_downstream_pressure = Pvap + R * (Pupstream - Pvap)

Cavitation begins when:

    downstream_pressure <= critical_downstream_pressure

The tracked cavitation margin is:

    cavitation_margin = critical_downstream_pressure - downstream_pressure

A positive margin means the venturi is cavitating. The throat pressure trace
also makes cavitation visible because it clips to vapor pressure once the
venturi is cavitating.
"""

import math

from fullflow import *
from fullplot import Axis, generate_map
from thermoprop import *


# -----------------------------------------------------------------------------
# Unit conversions and filenames
# -----------------------------------------------------------------------------

psi_to_pa = 6894.76
pa_to_psi = 1.0 / psi_to_pa

map_filename = "19cavitating_venturi"
results_filename = "19cavitating_venturi"

ambient_pressure = 14.7 * psi_to_pa


# -----------------------------------------------------------------------------
# Optional combustion-map generation
# -----------------------------------------------------------------------------
# Set this to True the first time the example is run if RP1_LOX.h5 does not
# already exist. Keep it False for normal transient runs so the example uses the
# existing HDF5 map instead of regenerating equilibrium chemistry every time.

generate_combustion_map = True

if generate_combustion_map:
    fuel = Propellant("rp-1", temperature=298.15)
    ox = Propellant("LOX", temperature=90.17)

    def rp1_lox_products_map(chamber_pressure, mixture_ratio):
        """Return mapped RP-1/LOX combustion-gas properties."""
        reactants = Reactants(
            fuels=fuel,
            oxidizers=ox,
            mixture_ratio=mixture_ratio,
        )

        equilibrium = Equilibrium(
            reactants=reactants,
            pressure=chamber_pressure,
        )

        return {
            "chamber_temperature": equilibrium.temperature,
            "gamma": equilibrium.gamma,
            "gas_constant": equilibrium.gas_constant,
        }

    generate_map(
        map_filename,
        group="products",
        axes=[
            Axis.linear("chamber_pressure", start=50 * psi_to_pa, stop=800 * psi_to_pa, count=20, units="Pa"),
            Axis.linear("mixture_ratio", start=0.8, stop=5.0, count=20),
        ],
        evaluate=rp1_lox_products_map,
        overwrite=True,
        raise_errors=True,
    )


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

FeedSystem = Network("Transient Cavitating Feed System")


# -----------------------------------------------------------------------------
# Source states
# -----------------------------------------------------------------------------
# The sources are fixed thermodynamic boundary states. The feed lines and nodes
# downstream of them are dynamic.

fuel_source_pressure = State(450.0 * psi_to_pa)
ox_source_pressure = State(450.0 * psi_to_pa)

FuelSource = Lookup(
    "Fuel Source",
    FeedSystem,
    Propellant,
    "rp-1",
    temperature=298.15,
    pressure=fuel_source_pressure,
)

OxSource = Lookup(
    "Oxidizer Source",
    FeedSystem,
    Propellant,
    "lox",
    temperature=90.17,
    pressure=ox_source_pressure,
)


# -----------------------------------------------------------------------------
# Primary pressure states
# -----------------------------------------------------------------------------
# These pressure states are owned by Volume components below. In steady state,
# FullFlow drives their mass derivatives to zero. In transient, FullFlow
# integrates the stored mass and solves pressure from the updated inventory.

fuel_venturi_upstream_pressure = State(448.0 * psi_to_pa)
fuel_manifold_pressure = State(435.0 * psi_to_pa)

ox_venturi_upstream_pressure = State(448.0 * psi_to_pa)
ox_manifold_pressure = State(435.0 * psi_to_pa)

chamber_pressure = State(150.0 * psi_to_pa)

# Initial guess only. This State is converted to a derived State after the
# injector mass flows exist so the combustion map sees the live mixture ratio.
mixture_ratio = State(2.5)


# -----------------------------------------------------------------------------
# Liquid property lookups for transient nodes
# -----------------------------------------------------------------------------
# These lookups give pressure-dependent density for each storage volume.

FuelVenturiUpstreamFluid = Lookup(
    "Fuel Venturi Upstream Fluid",
    FeedSystem,
    Propellant,
    FuelSource.composition,
    temperature=298.15,
    pressure=fuel_venturi_upstream_pressure,
)

FuelManifoldFluid = Lookup(
    "Fuel Manifold Fluid",
    FeedSystem,
    Propellant,
    FuelSource.composition,
    temperature=298.15,
    pressure=fuel_manifold_pressure,
)

OxVenturiUpstreamFluid = Lookup(
    "Ox Venturi Upstream Fluid",
    FeedSystem,
    Propellant,
    OxSource.composition,
    temperature=90.17,
    pressure=ox_venturi_upstream_pressure,
)

OxManifoldFluid = Lookup(
    "Ox Manifold Fluid",
    FeedSystem,
    Propellant,
    OxSource.composition,
    temperature=90.17,
    pressure=ox_manifold_pressure,
)


# -----------------------------------------------------------------------------
# Combustion-products map
# -----------------------------------------------------------------------------
# The map axes must match the input names used here:
#
#     chamber_pressure
#     mixture_ratio

ChamberMap = Map.from_hdf5(
    "Combustion Products Map",
    FeedSystem,
    map_filename,
    group="products",
    inputs={
        "chamber_pressure": chamber_pressure,
        "mixture_ratio": mixture_ratio,
    },
)

# Ideal-gas density estimate from the mapped chamber temperature and gas
# constant. This keeps the chamber as a simple mass-storage node.
chamber_density = chamber_pressure / (ChamberMap.gas_constant * ChamberMap.chamber_temperature)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

fuel_line_diameter = 0.75 / 39.37
ox_line_diameter = 0.75 / 39.37

fuel_line_area = (math.pi / 4.0) * fuel_line_diameter**2
ox_line_area = (math.pi / 4.0) * ox_line_diameter**2

fuel_line_length = 2.0
ox_line_length = 2.0

# The venturi throats are intentionally smaller than the fully-open injector
# effective areas. That lets the example transition from noncavitating to
# cavitating as injector demand rises.
fuel_venturi_throat_diameter = 0.22 / 39.37
ox_venturi_throat_diameter = 0.31 / 39.37

fuel_venturi_throat_area = (math.pi / 4.0) * fuel_venturi_throat_diameter**2
ox_venturi_throat_area = (math.pi / 4.0) * ox_venturi_throat_diameter**2

# The simple CavitatingVenturi component switches equations at incipient
# cavitation.  Choosing the cavitating Cd to match the noncavitating branch at
# the critical downstream pressure avoids an artificial mass-flow jump during
# the steady initialization and makes the transient cavitation onset easier to
# read.
venturi_pressure_recovery_factor = 0.85
venturi_noncavitating_cd = 0.60
venturi_cavitating_cd = venturi_noncavitating_cd * (1.0 - venturi_pressure_recovery_factor)**0.5

# Small but finite liquid volumes. Increasing these volumes slows the pressure
# response; decreasing them makes the transient stiffer and faster.
fuel_upstream_plenum_volume = 30.0e-6
fuel_manifold_volume = 40.0e-6

ox_upstream_plenum_volume = 30.0e-6
ox_manifold_volume = 40.0e-6

chamber_volume = (25.0 * 10.0) / (1550.0 * 39.37)


# -----------------------------------------------------------------------------
# Smooth injector-opening schedules
# -----------------------------------------------------------------------------
# The transient begins from the steady-state solution at t=0. Sequence is
# inactive during the steady solve, so each Cd target starts at the function
# value at t=0.


def smoothstep(x):
    """Smooth 0-to-1 transition with zero slope at both ends."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def smooth_ramp(t, start_time, end_time, start_value, end_value):
    """Smoothly ramp from start_value to end_value."""
    x = (t - start_time) / (end_time - start_time)
    s = smoothstep(x)
    return start_value + s * (end_value - start_value)


def fuel_injector_cd_schedule(t):
    """Fuel injector opens gradually from a nearly closed startup state."""
    return smooth_ramp(
        t,
        start_time=0.15,
        end_time=1.00,
        start_value=0.12,
        end_value=0.85,
    )


def ox_injector_cd_schedule(t):
    """Oxidizer injector opens later but faster than the fuel injector."""
    return smooth_ramp(
        t,
        start_time=0.30,
        end_time=0.80,
        start_value=0.12,
        end_value=0.95,
    )


FuelInjectorCd = Sequence(
    "Fuel Injector Cd Schedule",
    FeedSystem,
    function=fuel_injector_cd_schedule,
)

OxInjectorCd = Sequence(
    "Ox Injector Cd Schedule",
    FeedSystem,
    function=ox_injector_cd_schedule,
)


# -----------------------------------------------------------------------------
# Shared mass-flow states
# -----------------------------------------------------------------------------
# These are intentionally explicit so the flow path is easy to follow.

fuel_line_mass_flow = State(0.25)
fuel_venturi_mass_flow = State(0.25)
fuel_injector_mass_flow = State(0.25)
fuel_line_friction_factor = State(0.02)

ox_line_mass_flow = State(0.65)
ox_venturi_mass_flow = State(0.65)
ox_injector_mass_flow = State(0.65)
ox_line_friction_factor = State(0.02)


# -----------------------------------------------------------------------------
# Fuel feed line with inertia
# -----------------------------------------------------------------------------
# DarcyWeisbach is dynamic because it has finite length. Its mass_flow is
# integrated during the transient solve.

FuelLine = DarcyWeisbach(
    "Fuel Feed Line",
    FeedSystem,
    mass_flow=fuel_line_mass_flow,
    upstream_pressure=FuelSource.pressure,
    downstream_pressure=fuel_venturi_upstream_pressure,
    length=fuel_line_length,
    hydraulic_diameter=fuel_line_diameter,
    density=FuelSource.density,
    cross_sectional_area=fuel_line_area,
    friction_factor=fuel_line_friction_factor,
)

FuelLineFriction = Churchill(
    "Fuel Feed Line Friction",
    FeedSystem,
    mass_flow=FuelLine.mass_flow,
    friction_factor=FuelLine.friction_factor,
    hydraulic_diameter=FuelLine.hydraulic_diameter,
    dynamic_viscosity=FuelSource.dynamic_viscosity,
    cross_sectional_area=FuelLine.cross_sectional_area,
    roughness=1.0e-5,
)


# -----------------------------------------------------------------------------
# Oxidizer feed line with inertia
# -----------------------------------------------------------------------------

OxLine = DarcyWeisbach(
    "Ox Feed Line",
    FeedSystem,
    mass_flow=ox_line_mass_flow,
    upstream_pressure=OxSource.pressure,
    downstream_pressure=ox_venturi_upstream_pressure,
    length=ox_line_length,
    hydraulic_diameter=ox_line_diameter,
    density=OxSource.density,
    cross_sectional_area=ox_line_area,
    friction_factor=ox_line_friction_factor,
)

OxLineFriction = Churchill(
    "Ox Feed Line Friction",
    FeedSystem,
    mass_flow=OxLine.mass_flow,
    friction_factor=OxLine.friction_factor,
    hydraulic_diameter=OxLine.hydraulic_diameter,
    dynamic_viscosity=OxSource.dynamic_viscosity,
    cross_sectional_area=OxLine.cross_sectional_area,
    roughness=1.0e-5,
)


# -----------------------------------------------------------------------------
# Venturi upstream plenums
# -----------------------------------------------------------------------------
# These are transient storage nodes between the inertial feed lines and the
# cavitating venturis.

FuelVenturiUpstreamPlenum = Volume(
    "Fuel Venturi Upstream Plenum",
    FeedSystem,
    volume=fuel_upstream_plenum_volume,
    pressure=fuel_venturi_upstream_pressure,
    density=FuelVenturiUpstreamFluid.density,
    mass_flow_in=FuelLine.mass_flow,
    mass_flow_out=fuel_venturi_mass_flow,
)

OxVenturiUpstreamPlenum = Volume(
    "Ox Venturi Upstream Plenum",
    FeedSystem,
    volume=ox_upstream_plenum_volume,
    pressure=ox_venturi_upstream_pressure,
    density=OxVenturiUpstreamFluid.density,
    mass_flow_in=OxLine.mass_flow,
    mass_flow_out=ox_venturi_mass_flow,
)


# -----------------------------------------------------------------------------
# Cavitating venturis
# -----------------------------------------------------------------------------
# CavitatingVenturi is algebraic. It computes the flow allowed by the current
# upstream pressure, downstream pressure, vapor pressure, and throat area. The
# surrounding volumes and feed lines provide the transient storage and inertia.

FuelVenturi = CavitatingVenturi(
    "Fuel Cavitating Venturi",
    FeedSystem,
    upstream_pressure=FuelVenturiUpstreamPlenum.pressure,
    downstream_pressure=fuel_manifold_pressure,
    density=FuelVenturiUpstreamFluid.density,
    throat_area=fuel_venturi_throat_area,
    vapor_pressure=FuelSource.saturation_pressure,
    pressure_recovery_factor=venturi_pressure_recovery_factor,
    cavitating_discharge_coefficient=venturi_cavitating_cd,
    noncavitating_discharge_coefficient=venturi_noncavitating_cd,
    mass_flow=fuel_venturi_mass_flow,
)

OxVenturi = CavitatingVenturi(
    "Ox Cavitating Venturi",
    FeedSystem,
    upstream_pressure=OxVenturiUpstreamPlenum.pressure,
    downstream_pressure=ox_manifold_pressure,
    density=OxVenturiUpstreamFluid.density,
    throat_area=ox_venturi_throat_area,
    vapor_pressure=OxSource.saturation_pressure,
    pressure_recovery_factor=venturi_pressure_recovery_factor,
    cavitating_discharge_coefficient=venturi_cavitating_cd,
    noncavitating_discharge_coefficient=venturi_noncavitating_cd,
    mass_flow=ox_venturi_mass_flow,
)


# -----------------------------------------------------------------------------
# Injector manifolds
# -----------------------------------------------------------------------------
# These are transient storage nodes between the venturis and the injector
# orifices.

FuelManifold = Volume(
    "Fuel Injector Manifold",
    FeedSystem,
    volume=fuel_manifold_volume,
    pressure=fuel_manifold_pressure,
    density=FuelManifoldFluid.density,
    mass_flow_in=FuelVenturi.mass_flow,
    mass_flow_out=fuel_injector_mass_flow,
)

OxManifold = Volume(
    "Ox Injector Manifold",
    FeedSystem,
    volume=ox_manifold_volume,
    pressure=ox_manifold_pressure,
    density=OxManifoldFluid.density,
    mass_flow_in=OxVenturi.mass_flow,
    mass_flow_out=ox_injector_mass_flow,
)


# -----------------------------------------------------------------------------
# Injector orifices
# -----------------------------------------------------------------------------
# A small effective length gives the injector flow finite inertia. During steady
# state, FullFlow drives mass_flow_dot to zero. During transient, it integrates
# the injector mass-flow states.

FuelInjector = DischargeCoefficient(
    "Fuel Injector Orifices",
    FeedSystem,
    upstream_pressure=FuelManifold.pressure,
    downstream_pressure=chamber_pressure,
    density=FuelManifoldFluid.density,
    discharge_coefficient=FuelInjectorCd.target,
    cross_sectional_area=0.555e-4,
    length=0.04,
    mass_flow=fuel_injector_mass_flow,
)

OxInjector = DischargeCoefficient(
    "Ox Injector Orifices",
    FeedSystem,
    upstream_pressure=OxManifold.pressure,
    downstream_pressure=chamber_pressure,
    density=OxManifoldFluid.density,
    discharge_coefficient=OxInjectorCd.target,
    cross_sectional_area=1.25e-4,
    length=0.04,
    mass_flow=ox_injector_mass_flow,
)


# -----------------------------------------------------------------------------
# Mixture ratio from live injector flows
# -----------------------------------------------------------------------------
# This mutates the existing State object. ChamberMap already references
# mixture_ratio, so the map sees the live valve-coupled mixture ratio.

mixture_ratio <<= OxInjector.mass_flow / FuelInjector.mass_flow


# -----------------------------------------------------------------------------
# Combustion chamber and nozzle
# -----------------------------------------------------------------------------

Chamber = Volume(
    "Combustion Chamber",
    FeedSystem,
    volume=chamber_volume,
    pressure=chamber_pressure,
    density=chamber_density,
    mass_flow_in=FuelInjector.mass_flow + OxInjector.mass_flow,
)

Nozzle = IsentropicNozzle(
    "Nozzle",
    FeedSystem,
    upstream_total_pressure=Chamber.pressure,
    upstream_total_temperature=ChamberMap.chamber_temperature,
    ambient_pressure=ambient_pressure,
    specific_heat_ratio=ChamberMap.gamma,
    gas_constant=ChamberMap.gas_constant,
    throat_area=6.05 / 1550.0,
    expansion_ratio=4.5,
    mass_flow=Chamber.mass_flow_out,
)


# -----------------------------------------------------------------------------
# Cavitation diagnostic states
# -----------------------------------------------------------------------------
# Positive margin means the downstream pressure is below the venturi critical
# downstream pressure, so the venturi is cavitating.

FuelVenturiCriticalPressure = State._derived(
    lambda: getattr(FuelVenturi, "critical_downstream_pressure", FuelManifold.pressure.value)
)
FuelVenturiThroatPressure = State._derived(
    lambda: getattr(FuelVenturi, "throat_pressure", FuelVenturiUpstreamPlenum.pressure.value)
)
FuelCavitationMargin = (FuelVenturiCriticalPressure - FuelManifold.pressure) * pa_to_psi
FuelCavitating = State._derived(lambda: 1.0 if FuelCavitationMargin.value >= 0.0 else 0.0)

OxVenturiCriticalPressure = State._derived(
    lambda: getattr(OxVenturi, "critical_downstream_pressure", OxManifold.pressure.value)
)
OxVenturiThroatPressure = State._derived(
    lambda: getattr(OxVenturi, "throat_pressure", OxVenturiUpstreamPlenum.pressure.value)
)
OxCavitationMargin = (OxVenturiCriticalPressure - OxManifold.pressure) * pa_to_psi
OxCavitating = State._derived(lambda: 1.0 if OxCavitationMargin.value >= 0.0 else 0.0)


# -----------------------------------------------------------------------------
# Tracked outputs
# -----------------------------------------------------------------------------
# These are the most useful signals to plot from the HDF5 file.

FeedSystem.track("Fuel Source Pressure [psia]", FuelSource.pressure * pa_to_psi)
FeedSystem.track("Fuel Venturi Upstream Pressure [psia]", FuelVenturiUpstreamPlenum.pressure * pa_to_psi)
FeedSystem.track("Fuel Manifold Pressure [psia]", FuelManifold.pressure * pa_to_psi)
FeedSystem.track("Fuel Venturi Critical Downstream Pressure [psia]", FuelVenturiCriticalPressure * pa_to_psi)
FeedSystem.track("Fuel Venturi Throat Pressure [psia]", FuelVenturiThroatPressure * pa_to_psi)
FeedSystem.track("Fuel Cavitation Margin [psid]", FuelCavitationMargin)
FeedSystem.track("Fuel Cavitating Flag [-]", FuelCavitating)

FeedSystem.track("Ox Source Pressure [psia]", OxSource.pressure * pa_to_psi)
FeedSystem.track("Ox Venturi Upstream Pressure [psia]", OxVenturiUpstreamPlenum.pressure * pa_to_psi)
FeedSystem.track("Ox Manifold Pressure [psia]", OxManifold.pressure * pa_to_psi)
FeedSystem.track("Ox Venturi Critical Downstream Pressure [psia]", OxVenturiCriticalPressure * pa_to_psi)
FeedSystem.track("Ox Venturi Throat Pressure [psia]", OxVenturiThroatPressure * pa_to_psi)
FeedSystem.track("Ox Cavitation Margin [psid]", OxCavitationMargin)
FeedSystem.track("Ox Cavitating Flag [-]", OxCavitating)

FeedSystem.track("Fuel Injector Cd [-]", FuelInjectorCd.target)
FeedSystem.track("Ox Injector Cd [-]", OxInjectorCd.target)

FeedSystem.track("Fuel Line Mass Flow [kg/s]", FuelLine.mass_flow)
FeedSystem.track("Fuel Venturi Mass Flow [kg/s]", FuelVenturi.mass_flow)
FeedSystem.track("Fuel Injector Mass Flow [kg/s]", FuelInjector.mass_flow)

FeedSystem.track("Ox Line Mass Flow [kg/s]", OxLine.mass_flow)
FeedSystem.track("Ox Venturi Mass Flow [kg/s]", OxVenturi.mass_flow)
FeedSystem.track("Ox Injector Mass Flow [kg/s]", OxInjector.mass_flow)

FeedSystem.track("Chamber Pressure [psia]", Chamber.pressure * pa_to_psi)
FeedSystem.track("Chamber Temperature [K]", ChamberMap.chamber_temperature)
FeedSystem.track("Mixture Ratio [-]", mixture_ratio)
FeedSystem.track("Nozzle Mass Flow [kg/s]", Nozzle.mass_flow)


# -----------------------------------------------------------------------------
# Steady-state initialization
# -----------------------------------------------------------------------------
# This initializes the network at the t=0 injector Cd values. The t=0 injector
# values are intentionally small so the initial condition is a clean
# noncavitating steady state. The transient then opens the injectors smoothly
# and drives the venturis toward cavitation.

SteadyState(FeedSystem).solve(
    verbose=True,
    filename=results_filename,
)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------
# Use a modest timestep because the liquid nodes and short injector inertias are
# intentionally fast. Increase volumes or line lengths if you want a slower,
# smoother system response.

Transient(FeedSystem).solve(
    dt=0.002,
    t_final=1.20,
    verbose=True,
    statistics=True,
    filename=results_filename,
)
