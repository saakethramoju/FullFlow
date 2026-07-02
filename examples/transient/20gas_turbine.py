"""
Transient Turbopump Startup: Combustion Map, Turbine Map, and Rotor
===================================================================

This example is the first transient turbopump-building block for a later
oxygen-rich powerpack model. It starts with only the gas generator / preburner,
the oxygen-rich turbine, and the common rotor. The LOX pump will be added in the
next example step.

The purpose of this file is to demonstrate a clean startup transient from
approximately zero shaft speed:

    1. A gas-generator pressure command ramps up smoothly.
    2. A preburner combustion-products map supplies turbine gas properties.
    3. A startup-safe turbine map supplies turbine flow parameter and torque.
    4. The Rotor component integrates shaft speed from the mapped torque.

Physical Layout
---------------

                       CH4 / O2 IGNITER PRODUCTS
                                  |
                                  v

                         OXYGEN-RICH PREBURNER
                  +--------------------------------+
                  | LNG + LOX + igniter products   |
                  | combustion-products map        |
                  +---------------+----------------+
                                  |
                                  | hot oxygen-rich turbine gas
                                  v
                         +----------------------+
                         | TURBINE INLET        |
                         | Pt(t), Tt(map), gas  |
                         +----------+-----------+
                                    |
                                    v
                         /======================\\
                        ||   OXYGEN-RICH        ||
                        ||      TURBINE         ||
                         \\======================//
                                    |
                                    | turbine torque from map
                                    v

        ==============================================================
        ||                  COMMON ROTOR SHAFT                       ||
        ||                                                          ||
        ||      net torque ---> rotor inertia ---> shaft speed       ||
        ==============================================================

Startup Map Behavior
--------------------

The turbine map includes the zero-speed startup region. It maps torque directly,
not shaft power. This avoids the usual singularity:

    torque = power / omega

at omega = 0.

For the turbine-only startup in this file, a small temporary shaft-drag load is
used to give the rotor a steady endpoint before the pump map is added. In the
next step, the rotor net torque will be changed to:

    net_torque = turbine_torque - pump_torque

and the pump map will provide the real load.

Notes
-----

The numerical values are Raptor-oxygen-powerpack-inspired demonstration values.
They are not calibrated SpaceX hardware data. The map shapes are synthetic and
are intended to be smooth, finite, and startup-safe.
"""

import numpy as np

from fullflow import *
from fullplot import Axis, generate_map
from thermoprop import *


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Turbopump = Network("Transient Turbopump Startup")


# -----------------------------------------------------------------------------
# Units
# -----------------------------------------------------------------------------

bar = 100000.0


# -----------------------------------------------------------------------------
# Design / reference point
# -----------------------------------------------------------------------------
# These reference values set the scaling for the synthetic preburner and turbine
# maps. They are chosen to be representative of a high-pressure oxygen-rich
# powerpack, but they are not hardware-calibrated data.

OxRichPreburnerDesignPressure = 600.0 * bar
OxRichTurbineDownstreamPressure = 350.0 * bar
OxRichPreburnerMixtureRatio = 85.0

FreeRunRotorSpeed = 36000.0                 # rpm, turbine-only asymptotic speed
ReferenceTurbinePower = 30.0e6              # W, demonstration scaling
ReferenceTurbineEfficiency = 0.55

RotorPolarMomentOfInertia = 1.0             # kg*m^2, demonstration shaft inertia


# -----------------------------------------------------------------------------
# Map file settings
# -----------------------------------------------------------------------------
# This file contains two maps:
#
#     /preburner_products
#         inputs:  preburner_pressure, mixture_ratio
#         outputs: preburner_temperature, gamma, gas_constant, enthalpy
#
#     /oxygen_rich_turbine
#         inputs:  upstream_total_pressure, downstream_pressure,
#                  upstream_total_temperature, rotor_speed
#         outputs: flow_parameter, torque, ideal_total_enthalpy_change
#
# Set force_regenerate_maps=True when changing the map definitions.

filename = "20gas_turbine"

# Keep this True while developing/changing the map definitions. After the map
# exists and the definitions are stable, set it to False to skip regeneration.
generate_maps = True


# -----------------------------------------------------------------------------
# Igniter reference products
# -----------------------------------------------------------------------------
# The igniter is only used while generating the preburner-products map. It gives
# the preburner reactants the same flavor as the steady-state turbopump example.

IgniterMethaneSource = CombustionGas(
    fluid="ch4",
    temperature=300.0,
)

IgniterGOXSource = CombustionGas(
    fluid="ox",
    temperature=300.0,
)

IgniterReactants = Reactants(
    fuels=IgniterMethaneSource,
    oxidizers=IgniterGOXSource,
    mixture_ratio=3.0,
)

IgniterChamber = Equilibrium(
    reactants=IgniterReactants,
    pressure=OxRichPreburnerDesignPressure,
)


# -----------------------------------------------------------------------------
# Map generation
# -----------------------------------------------------------------------------
# The transient solve reads from HDF5 maps. The expensive equilibrium chemistry is
# done here once during map generation, not during every nonlinear evaluation.

PreburnerFuel = Propellant("lng", temperature=111.0)
PreburnerOxidizer = Propellant("lox", temperature=90.0)


def clamp(x, lower, upper):
    """Clamp x to the closed interval [lower, upper]."""
    return max(lower, min(upper, x))


def preburner_products_map(preburner_pressure, mixture_ratio):
    """Return oxygen-rich preburner products for one map grid point."""
    reactants = Reactants(
        fuels=PreburnerFuel,
        oxidizers=PreburnerOxidizer,
        igniters=IgniterChamber.CombustionGas,
        igniter_fraction=0.02,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=reactants,
        pressure=preburner_pressure,
    )

    return {
        "preburner_temperature": eq.temperature,
        "gamma": eq.gamma,
        "gas_constant": eq.gas_constant,
        "enthalpy": eq.enthalpy,
    }


# Create one design preburner state so the synthetic turbine map can be scaled
# from a reasonable gas temperature, gas constant, and gamma.
DesignPreburnerReactants = Reactants(
    fuels=PreburnerFuel,
    oxidizers=PreburnerOxidizer,
    igniters=IgniterChamber.CombustionGas,
    igniter_fraction=0.02,
    mixture_ratio=OxRichPreburnerMixtureRatio,
)

DesignPreburner = Equilibrium(
    reactants=DesignPreburnerReactants,
    pressure=OxRichPreburnerDesignPressure,
)

DesignTurbineGasConstant = DesignPreburner.gas_constant
DesignTurbineGamma = DesignPreburner.gamma
DesignTurbineTemperature = DesignPreburner.temperature
DesignTurbineCp = DesignTurbineGamma * DesignTurbineGasConstant / (DesignTurbineGamma - 1.0)
DesignPressureRatio = OxRichPreburnerDesignPressure / OxRichTurbineDownstreamPressure
DesignIdealEnthalpyChange = DesignTurbineCp * DesignTurbineTemperature * (1.0 - (1.0 / DesignPressureRatio) ** ((DesignTurbineGamma - 1.0) / DesignTurbineGamma))
DesignTurbineMassFlow = ReferenceTurbinePower / (ReferenceTurbineEfficiency * DesignIdealEnthalpyChange)
DesignRotorOmega = (np.pi / 30.0) * FreeRunRotorSpeed
DesignTurbineTorque = ReferenceTurbinePower / DesignRotorOmega


def pressure_drive(upstream_total_pressure, downstream_pressure):
    """
    Dimensionless turbine pressure-drive factor.

    Returns 0 when pressure ratio is not useful and 1 at the design pressure
    ratio. Values above 1 are allowed slightly so the map can represent a little
    startup oversupply.
    """
    if upstream_total_pressure <= downstream_pressure:
        return 0.0

    pressure_ratio = upstream_total_pressure / downstream_pressure
    available = 1.0 - 1.0 / pressure_ratio
    design_available = 1.0 - 1.0 / DesignPressureRatio

    return clamp(available / design_available, 0.0, 1.25)


def oxygen_rich_turbine_map(
    upstream_total_pressure,
    downstream_pressure,
    upstream_total_temperature,
    rotor_speed,
):
    """
    Startup-safe oxygen-rich turbine map.

    The map returns torque directly so zero rotor speed is allowed. The returned
    flow parameter has a tiny numerical floor so the current GasTurbine component
    can evaluate efficiency and discharge enthalpy without dividing by exactly
    zero mass flow at the first startup point.
    """
    drive = pressure_drive(upstream_total_pressure, downstream_pressure)

    speed_ratio = rotor_speed / FreeRunRotorSpeed
    speed_ratio = max(speed_ratio, 0.0)

    temperature_factor = (DesignTurbineTemperature / upstream_total_temperature) ** 0.5

    # Flow is mostly pressure-ratio driven. The speed correction is deliberately
    # weak because this is a simple demonstration turbine map.
    mass_flow = DesignTurbineMassFlow
    mass_flow = mass_flow * drive**0.5
    mass_flow = mass_flow * (upstream_total_pressure / OxRichPreburnerDesignPressure)
    mass_flow = mass_flow * temperature_factor
    mass_flow = mass_flow * clamp(1.02 - 0.05 * speed_ratio, 0.90, 1.05)

    # Keep a tiny numerical floor. This behaves like no-flow for plotting, but
    # avoids a divide-by-zero inside GasTurbine at t = 0.
    mass_flow = max(mass_flow, 1.0e-8 * DesignTurbineMassFlow)

    flow_parameter = mass_flow * np.sqrt(DesignTurbineGasConstant * upstream_total_temperature) / upstream_total_pressure

    # Turbine torque is mapped directly and remains finite at zero speed.
    # The speed correction is mild; the rotor settles because of the temporary
    # shaft-drag load below, not because the turbine torque is artificially
    # forced to zero at the free-running speed.
    speed_torque_factor = clamp(1.10 - 0.10 * speed_ratio, 0.85, 1.10)

    torque = DesignTurbineTorque
    torque = torque * drive
    torque = torque * temperature_factor
    torque = torque * speed_torque_factor

    # The ideal total enthalpy change is included as a mapped output so
    # GasTurbine does not need to divide by a zero ideal drop at pressure ratio 1.
    if upstream_total_pressure > downstream_pressure:
        pressure_ratio = upstream_total_pressure / downstream_pressure
        ideal_total_enthalpy_change = DesignTurbineCp * upstream_total_temperature * (1.0 - (1.0 / pressure_ratio) ** ((DesignTurbineGamma - 1.0) / DesignTurbineGamma))
    else:
        ideal_total_enthalpy_change = 0.0

    ideal_total_enthalpy_change = max(ideal_total_enthalpy_change, 1.0)

    return {
        "flow_parameter": flow_parameter,
        "torque": torque,
        "ideal_total_enthalpy_change": ideal_total_enthalpy_change,
    }


if generate_maps:
    generate_map(
        filename,
        group="preburner_products",
        axes=[
            Axis.values(
                "preburner_pressure",
                [
                    350.0 * bar,
                    400.0 * bar,
                    500.0 * bar,
                    600.0 * bar,
                    700.0 * bar,
                ],
                units="Pa",
            ),
            Axis.values(
                "mixture_ratio",
                [
                    60.0,
                    75.0,
                    85.0,
                    100.0,
                    120.0,
                ],
            ),
        ],
        evaluate=preburner_products_map,
        outputs=[
            "preburner_temperature",
            "gamma",
            "gas_constant",
            "enthalpy",
        ],
        overwrite=True,
        raise_errors=True,
    )

    generate_map(
        filename,
        group="oxygen_rich_turbine",
        axes=[
            Axis.values(
                "upstream_total_pressure",
                [
                    350.0 * bar,
                    400.0 * bar,
                    500.0 * bar,
                    600.0 * bar,
                    700.0 * bar,
                ],
                units="Pa",
            ),
            Axis.values(
                "downstream_pressure",
                [
                    300.0 * bar,
                    350.0 * bar,
                    400.0 * bar,
                ],
                units="Pa",
            ),
            Axis.values(
                "upstream_total_temperature",
                [
                    0.70 * DesignTurbineTemperature,
                    0.80 * DesignTurbineTemperature,
                    0.90 * DesignTurbineTemperature,
                    1.00 * DesignTurbineTemperature,
                    1.10 * DesignTurbineTemperature,
                ],
                units="K",
            ),
            Axis.values(
                "rotor_speed",
                [
                    0.0,
                    500.0,
                    1000.0,
                    3000.0,
                    6000.0,
                    12000.0,
                    18000.0,
                    24000.0,
                    30000.0,
                    36000.0,
                    42000.0,
                ],
                units="rpm",
            ),
        ],
        evaluate=oxygen_rich_turbine_map,
        outputs=[
            "flow_parameter",
            "torque",
            "ideal_total_enthalpy_change",
        ],
        overwrite=True,
        raise_errors=True,
    )


# -----------------------------------------------------------------------------
# Smooth gas-generator startup command
# -----------------------------------------------------------------------------
# The pressure command starts nearly at turbine downstream pressure so the
# turbine begins with essentially no pressure ratio. It then ramps to the design
# preburner pressure.


def smoothstep(x):
    """Smooth 0-to-1 transition with zero slope at both ends."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def smooth_ramp(t, start_time, end_time, start_value, end_value):
    """Smoothly ramp from start_value to end_value."""
    x = (t - start_time) / (end_time - start_time)
    s = smoothstep(x)

    return start_value + s * (end_value - start_value)


def preburner_pressure_schedule(t):
    """Gas-generator / preburner pressure ramp."""
    return smooth_ramp(
        t,
        start_time=0.05,
        end_time=1.00,
        start_value=350.0 * bar,
        end_value=OxRichPreburnerDesignPressure,
    )


def preburner_mixture_ratio_schedule(t):
    """
    Oxygen-rich preburner mixture-ratio ramp.

    At fixed inlet propellant temperatures, this map is much more sensitive to
    mixture ratio than pressure.  Starting very oxygen-rich gives a colder
    initial turbine gas.  Ramping toward the design MR gives the preburner
    temperature a visible, physically motivated startup transient.
    """
    return smooth_ramp(
        t,
        start_time=0.05,
        end_time=1.00,
        start_value=120.0,
        end_value=OxRichPreburnerMixtureRatio,
    )


# -----------------------------------------------------------------------------
# Transient states
# -----------------------------------------------------------------------------

PreburnerPressure = State(350.0 * bar)
PreburnerMixtureRatio = State(120.0)
TurbineDownstreamPressure = State(OxRichTurbineDownstreamPressure)
RotorSpeed = State(0.0, bounds=(0.0, 42000.0))

PreburnerPressureCommand = Sequence(
    "Preburner Pressure Command",
    Turbopump,
    target=PreburnerPressure,
    function=preburner_pressure_schedule,
)

PreburnerMixtureRatioCommand = Sequence(
    "Preburner Mixture Ratio Command",
    Turbopump,
    target=PreburnerMixtureRatio,
    function=preburner_mixture_ratio_schedule,
)


# -----------------------------------------------------------------------------
# Combustion-products map
# -----------------------------------------------------------------------------
# The preburner map is the transient combustion model. It turns the commanded
# preburner pressure and mixture ratio into turbine-gas properties.

PreburnerMap = Map.from_hdf5(
    "Preburner Products Map",
    Turbopump,
    filename=filename,
    group="preburner_products",
    inputs={
        "preburner_pressure": PreburnerPressure,
        "mixture_ratio": PreburnerMixtureRatio,
    },
    outputs=[
        "preburner_temperature",
        "gamma",
        "gas_constant",
        "enthalpy",
    ],
)


# -----------------------------------------------------------------------------
# Turbine map
# -----------------------------------------------------------------------------
# This startup map is intentionally defined at rotor_speed = 0 so the transient
# can start from rest. It returns torque directly, so no power/omega singularity
# appears in the map.

TurbineMap = Map.from_hdf5(
    "Oxygen-Rich Turbine Map",
    Turbopump,
    filename=filename,
    group="oxygen_rich_turbine",
    inputs={
        "upstream_total_pressure": PreburnerPressure,
        "downstream_pressure": TurbineDownstreamPressure,
        "upstream_total_temperature": PreburnerMap.preburner_temperature,
        "rotor_speed": RotorSpeed,
    },
    outputs=[
        "flow_parameter",
        "torque",
        "ideal_total_enthalpy_change",
    ],
)


# -----------------------------------------------------------------------------
# Gas turbine component
# -----------------------------------------------------------------------------
# GasTurbine converts the mapped flow parameter into mass flow and converts the
# mapped torque into shaft power:
#
#     mdot = FP * Pt / sqrt(R*Tt)
#     shaft_power = torque * omega

OxygenRichTurbine = GasTurbine(
    "Oxygen-Rich Turbine",
    Turbopump,
    rotor_speed=RotorSpeed,
    torque=TurbineMap.torque,
    flow_parameter=TurbineMap.flow_parameter,
    upstream_total_pressure=PreburnerPressure,
    upstream_total_temperature=PreburnerMap.preburner_temperature,
    downstream_pressure=TurbineDownstreamPressure,
    gas_constant=PreburnerMap.gas_constant,
    specific_heat_ratio=PreburnerMap.gamma,
    upstream_total_enthalpy=PreburnerMap.enthalpy,
    ideal_total_enthalpy_change=TurbineMap.ideal_total_enthalpy_change,
)


# -----------------------------------------------------------------------------
# Rotor dynamics
# -----------------------------------------------------------------------------
# This first turbine-only example has no pump load yet.  To keep the rotor from
# accelerating forever, add a small temporary shaft-drag load that grows with
# speed and balances the demonstration turbine near FreeRunRotorSpeed.
#
# In the pump version, replace this temporary load with the real pump torque:
#
#     RotorNetTorque = TurbineMap.torque - LOXPumpMap.torque

RotorDragTorque = DesignTurbineTorque * (RotorSpeed / FreeRunRotorSpeed) ** 2
RotorNetTorque = TurbineMap.torque - RotorDragTorque

TurbopumpRotor = Rotor(
    "Turbopump Rotor",
    Turbopump,
    rotor_speed=RotorSpeed,
    polar_moment_of_inertia=RotorPolarMomentOfInertia,
    net_torque=RotorNetTorque,
)


# -----------------------------------------------------------------------------
# Tracked outputs
# -----------------------------------------------------------------------------

Turbopump.track("Preburner Pressure [bar]", PreburnerPressure / bar)
Turbopump.track("Turbine Downstream Pressure [bar]", TurbineDownstreamPressure / bar)
Turbopump.track("Turbine Pressure Ratio", PreburnerPressure / TurbineDownstreamPressure)
Turbopump.track("Preburner Mixture Ratio", PreburnerMixtureRatio)
Turbopump.track("Preburner Temperature [K]", PreburnerMap.preburner_temperature)
Turbopump.track("Preburner Gamma [-]", PreburnerMap.gamma)
Turbopump.track("Preburner Gas Constant [J/kg/K]", PreburnerMap.gas_constant)

Turbopump.track("Rotor Speed [rpm]", RotorSpeed)
Turbopump.track("Turbine Torque [N*m]", TurbineMap.torque)
Turbopump.track("Temporary Rotor Drag Torque [N*m]", RotorDragTorque)
Turbopump.track("Rotor Net Torque [N*m]", RotorNetTorque)
Turbopump.track("Turbine Mass Flow [kg/s]", OxygenRichTurbine.mass_flow)
Turbopump.track("Turbine Flow Parameter", TurbineMap.flow_parameter)
Turbopump.track("Turbine Shaft Power [W]", OxygenRichTurbine.shaft_power)
Turbopump.track("Turbine Shaft Efficiency [-]", OxygenRichTurbine.efficiency)
Turbopump.track("Ideal Total Enthalpy Change [J/kg]", TurbineMap.ideal_total_enthalpy_change)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------
# Do not run a normal steady-state solve before this transient. The point of this
# example is to start from 0 rpm and let the rotor accelerate from the transient
# gas-generator startup command.

Transient(Turbopump).solve(
    dt=0.002,
    t_final=3.0,
    verbose=True,
    statistics=True,
    filename=filename,
)
