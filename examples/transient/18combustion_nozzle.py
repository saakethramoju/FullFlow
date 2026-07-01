"""
Smooth transient feed-system throttling example.

Physical layout
---------------

                         Fuel tank / source
                     RP-1, 350 psia, 298.15 K
                                |
                                v
                         +-------------+
                         | Fuel valve  |  Cd_f(t)
                         +-------------+
                                |
                                v
    +---------------------------------------------------------------+
    |                                                               |
    |                    Combustion chamber                          |
    |             equilibrium gas, finite gas volume                  |
    |                                                               |
    +---------------------------------------------------------------+
                                |
                                v
                         +-------------+
                         |   Nozzle    |
                         +-------------+
                                |
                                v
                         Ambient pressure
                             14.7 psia

                         Oxidizer tank / source
                       LOX, 350 psia, saturated liquid
                                |
                                v
                         +-------------+
                         | Ox valve    |  Cd_ox(t)
                         +-------------+
                                |
                                v
                         Combustion chamber

Model description
-----------------
This example models a simple pressure-fed RP-1/LOX feed system connected to a
combustion chamber and nozzle.

The fuel and oxidizer valves are represented with incompressible discharge
coefficient components. Their discharge coefficients are prescribed by smooth
functions of time. The oxidizer valve starts closing later than the fuel valve,
but closes over a shorter time interval, so the oxidizer-to-fuel mass-flow ratio
changes during the transient.

The propellant mixture ratio is not fixed. It is computed from the valve mass
flows:

    mixture_ratio = oxidizer_mass_flow / fuel_mass_flow

That mixture ratio is passed into the Reactants lookup. The chamber gas is then
computed from an equilibrium lookup using the current chamber pressure and
mixture ratio.

The chamber is modeled as a finite gas volume with mass storage. The nozzle mass
flow is computed from the current chamber state, and the chamber pressure evolves
from the difference between total propellant inflow and nozzle outflow.

This is intentionally a compact user example. It demonstrates:

    1. Lookup components for propellant and combustion-gas properties
    2. Function-based Sequence components for smooth transient inputs
    3. A finite chamber Volume coupled to an IsentropicNozzle
    4. A steady-state initialization followed by a transient solve
"""

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Unit conversions and constants
# -----------------------------------------------------------------------------

psi_to_pa = 6894.76

ambient_pressure = 14.7 * psi_to_pa


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

FeedSystem = Network("Feed System")


# -----------------------------------------------------------------------------
# Propellant source states
# -----------------------------------------------------------------------------
#
# These sources are treated as fixed-pressure reservoirs. The Lookup components
# evaluate ThermoProp objects, and their properties can then be used directly by
# FullFlow components.

FuelSource = Lookup(
    "Fuel Source",
    FeedSystem,
    Propellant,
    "rp-1",
    temperature=298.15,
    pressure=350.0 * psi_to_pa,
)

OxSource = Lookup(
    "Oxidizer Source",
    FeedSystem,
    Propellant,
    "lox",
    pressure=350.0 * psi_to_pa,
    quality=0.0,
)


# -----------------------------------------------------------------------------
# Combustion reactants and equilibrium gas
# -----------------------------------------------------------------------------
#
# The initial mixture_ratio value is only an initial guess. Later in the script,
# Props.mixture_ratio is reassigned to the actual valve flow ratio:
#
#     OxValve.mass_flow / FuelValve.mass_flow
#
# The steady-state solve brings this value to the flow-consistent initial state
# before the transient solve begins.

Props = Lookup(
    "Propellants",
    FeedSystem,
    Reactants,
    fuels=Propellant("rp-1", temperature=298.15),
    oxidizers=Propellant("lox", temperature=90.17),
    mixture_ratio=2.5,
)

ChamberGas = Lookup(
    "Chamber Gas",
    FeedSystem,
    Equilibrium,
    reactants=Props,
    pressure=300.0 * psi_to_pa,
)


# -----------------------------------------------------------------------------
# Smooth valve schedules
# -----------------------------------------------------------------------------
#
# Sequence can use a function instead of a table of time/value points. The
# function receives the current transient time and returns the scheduled value.
#
# Here both valve schedules are smooth ramps. The fuel valve closes slowly over
# most of the transient, while the oxidizer valve starts closing later but closes
# faster. This makes the chamber pressure and mixture ratio change smoothly.


def smoothstep(x):
    """Smooth 0-to-1 transition with zero slope at both endpoints."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def smooth_ramp(t, start_time, end_time, start_value, end_value):
    """Smoothly ramp from start_value to end_value between two times."""
    x = (t - start_time) / (end_time - start_time)
    s = smoothstep(x)

    return start_value + s * (end_value - start_value)


def fuel_cd_schedule(t):
    """
    Fuel valve schedule.

    The fuel valve holds fully open until 0.10 s, then smoothly closes from
    Cd = 1.00 to Cd = 0.55 by 1.10 s.
    """
    return smooth_ramp(
        t,
        start_time=0.10,
        end_time=1.10,
        start_value=1.00,
        end_value=0.55,
    )


def ox_cd_schedule(t):
    """
    Oxidizer valve schedule.

    The oxidizer valve holds fully open until 0.20 s, then smoothly closes from
    Cd = 1.00 to Cd = 0.65 by 0.80 s.
    """
    return smooth_ramp(
        t,
        start_time=0.20,
        end_time=0.80,
        start_value=1.00,
        end_value=0.65,
    )


FuelCd = Sequence(
    "Fuel Cd Schedule",
    FeedSystem,
    function=fuel_cd_schedule,
)

OxCd = Sequence(
    "Oxidizer Cd Schedule",
    FeedSystem,
    function=ox_cd_schedule,
)


# -----------------------------------------------------------------------------
# Feed valves
# -----------------------------------------------------------------------------
#
# The discharge-coefficient components compute incompressible flow from the
# source pressures to the chamber pressure. The chamber pressure comes from the
# chamber gas lookup / volume state, so the valve flow rates respond naturally as
# chamber pressure changes.

FuelValve = DischargeCoefficient(
    "Fuel Valve",
    FeedSystem,
    upstream_pressure=FuelSource.pressure,
    downstream_pressure=ChamberGas.pressure,
    density=FuelSource.density,
    discharge_coefficient=FuelCd.target,
    cross_sectional_area=0.9e-4,
)

OxValve = DischargeCoefficient(
    "Oxidizer Valve",
    FeedSystem,
    upstream_pressure=OxSource.pressure,
    downstream_pressure=ChamberGas.pressure,
    density=OxSource.density,
    discharge_coefficient=OxCd.target,
    cross_sectional_area=1.5e-4,
)


# The combustion mixture ratio is determined by the feed system, not prescribed.
Props.mixture_ratio = OxValve.mass_flow / FuelValve.mass_flow


# -----------------------------------------------------------------------------
# Combustion chamber volume
# -----------------------------------------------------------------------------
#
# The chamber stores gas mass. The current chamber pressure and gas density come
# from the equilibrium gas lookup. The chamber receives fuel + oxidizer inflow
# and discharges through the nozzle.

Chamber = Volume(
    "Combustion Chamber",
    FeedSystem,
    volume=(25.0 * 10.0) / (1550.0 * 39.37),
    pressure=ChamberGas.pressure,
    density=ChamberGas.density,
    mass_flow_in=FuelValve.mass_flow + OxValve.mass_flow,
)


# -----------------------------------------------------------------------------
# Nozzle
# -----------------------------------------------------------------------------
#
# The nozzle outflow is tied to Chamber.mass_flow_out, which closes the chamber
# mass balance. As the valve schedules reduce propellant flow, the chamber
# pressure changes until nozzle outflow again balances inflow.

Nozzle = IsentropicNozzle(
    "Nozzle",
    FeedSystem,
    upstream_total_pressure=Chamber.pressure,
    upstream_total_temperature=ChamberGas.temperature,
    ambient_pressure=ambient_pressure,
    specific_heat_ratio=ChamberGas.gamma,
    gas_constant=ChamberGas.gas_constant,
    throat_area=6.5 / 1550.0,
    expansion_ratio=5.0,
    mass_flow=Chamber.mass_flow_out,
)


# -----------------------------------------------------------------------------
# Tracked outputs
# -----------------------------------------------------------------------------
#
# These are written to the HDF5 file during both the steady-state and transient
# solves. Tracking the valve commands and mass flows makes it easy to interpret
# why chamber pressure and mixture ratio move the way they do.

FeedSystem.track("Chamber Pressure [psia]", Chamber.pressure / psi_to_pa)
FeedSystem.track("Mixture Ratio", Props.mixture_ratio)

FeedSystem.track("Fuel Cd", FuelCd.target)
FeedSystem.track("Oxidizer Cd", OxCd.target)

FeedSystem.track("Fuel Mass Flow [kg/s]", FuelValve.mass_flow)
FeedSystem.track("Oxidizer Mass Flow [kg/s]", OxValve.mass_flow)
FeedSystem.track("Total Propellant Mass Flow [kg/s]", FuelValve.mass_flow + OxValve.mass_flow)


# -----------------------------------------------------------------------------
# Steady-state initialization
# -----------------------------------------------------------------------------
#
# The steady-state solve finds a consistent initial chamber pressure, valve flow
# rates, nozzle flow rate, and mixture ratio before the transient starts.

filename = "18combustion_nozzle"

SteadyState(FeedSystem).solve(
    verbose=True,
    filename=filename,
)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------
#
# The transient solve starts from the converged steady-state solution. The valve
# Cd functions are then evaluated at each timestep, changing the feed-system
# operating point smoothly over time.

Transient(FeedSystem).solve(
    dt=0.005,
    t_final=1.1,
    verbose=True,
    statistics=True,
    filename=filename,
)
