"""
Oxygen-Rich Preburner / LOX Turbopump Example
=============================================

This example models a simplified oxygen-rich preburner driving a LOX
turbopump. The turbine and pump are connected by a common rotor shaft, and the
steady-state solver adjusts shaft speed until the turbine torque balances the
pump torque.

The example demonstrates:

- Equilibrium combustion products from an oxygen-rich preburner
- Turbine performance from a map
- LOX pump performance from a map
- Shaft coupling through a shared rotor speed
- Torque balance using a Rotor component

Physical Layout
---------------

                             IGNITER
                   +--------------------------+
                   |  CH4 / O2 Igniter Gas    |
                   +------------+-------------+
                                |
                                v

                        OXYGEN-RICH PREBURNER
                 +----------------------------------+
                 |  LNG + LOX + Igniter Products    |
                 |  Equilibrium Combustion Source   |
                 +----------------+-----------------+
                                  |
                                  | hot oxygen-rich turbine gas
                                  v
                         +----------------------+
                         |  TURBINE INLET       |
                         |  Total Pressure, Tt  |
                         +----------+-----------+
                                    |
                                    v
                         /======================\\
                        ||   OXYGEN-RICH        ||
                        ||      TURBINE         ||
                         \\======================//
                                    |
                                    | shaft power extracted
                                    v

     ================================================================
     ||                     COMMON ROTOR SHAFT                     ||
     ||                                                            ||
     ||     turbine torque  --->   SHAFT SPEED   --->  pump load   ||
     ================================================================
                                    ^
                                    |
                                    | mechanical coupling
                                    |

                         /======================\\
                        ||       LOX PUMP       ||
                         \\======================//
                                    |
                                    | pressurized liquid oxygen
                                    v
                      +-------------------------------+
                      |  HIGH-PRESSURE LOX DISCHARGE  |
                      +-------------------------------+

     LOX SUPPLY
+-------------------+
|  Low-Pressure LOX |
+---------+---------+
          |
          | liquid oxygen
          v
   +-------------+
   | Pump Inlet  |
   +-------------+

Governing Coupling
------------------

The turbine and pump are linked by the same shaft speed:

    turbine rotor speed = pump rotor speed

The rotor enforces steady-state shaft torque balance:

    turbine torque - pump torque = 0

The LOX pump map returns:

    head_rise
    torque

The turbine map returns:

    flow_parameter
    torque
    ideal_total_enthalpy_change

The pump head is converted internally into discharge pressure using:

    ΔP = ρ g H

so a large head rise in meters corresponds to a very large pressure rise in Pa.

Notes
-----

This is a simplified demonstration model intended to show how FullFlow
components can be coupled together. The turbine and pump maps are smooth,
user-generated performance maps rather than detailed hardware-calibrated maps.
"""

import numpy as np

from fullflow import *
from fullplot import Axis, generate_map
from thermoprop import *


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Turbopump = Network("Turbopump")


# -----------------------------------------------------------------------------
# Units
# -----------------------------------------------------------------------------

bar = 100000.0


# -----------------------------------------------------------------------------
# Design point
# -----------------------------------------------------------------------------
# These values set the nominal operating point for the simplified oxygen-rich
# preburner and LOX turbopump.

OxRichPreburnerPressure = 600.0 * bar
OxRichTurbineDownstreamPressure = 350.0 * bar
OxRichTurbineRotorSpeed = 36000.0

LOXPumpDesignMassFlow = 470.0
LOXPumpDesignInletPressure = 5.0 * bar
LOXPumpDesignDischargePressure = 550.0 * bar
LOXPumpDesignRotorSpeed = OxRichTurbineRotorSpeed

LOXPumpDesignEfficiency = 0.72
OxRichTurbineDesignEfficiency = 0.55


# -----------------------------------------------------------------------------
# Igniter
# -----------------------------------------------------------------------------
# The igniter is represented as a small methane/oxygen combustion source.
# Its product gas is fed into the preburner as an igniter stream.

IgniterMethaneSource = CombustionGas(
    fluid="ch4",
    temperature=300,
)

IgniterGOXSource = CombustionGas(
    fluid="ox",
    temperature=300,
)

IgniterReactants = Reactants(
    fuels=IgniterMethaneSource,
    oxidizers=IgniterGOXSource,
    mixture_ratio=3,
)

IgniterChamber = Equilibrium(
    reactants=IgniterReactants,
    pressure=OxRichPreburnerPressure,
)


# -----------------------------------------------------------------------------
# Oxygen-rich preburner
# -----------------------------------------------------------------------------
# The preburner is intentionally very oxygen-rich. Its gas composition is used
# to build the turbine map and to evaluate the GasTurbine component.

PreburnerReactants = Reactants(
    fuels=Propellant("lng", temperature=111),
    oxidizers=Propellant("lox", temperature=90),
    igniters=IgniterChamber.CombustionGas,
    igniter_fraction=0.02,
    mixture_ratio=85,
)

Preburner = Equilibrium(
    reactants=PreburnerReactants,
    pressure=OxRichPreburnerPressure,
)


# -----------------------------------------------------------------------------
# LOX pump fluid
# -----------------------------------------------------------------------------
# The LOX pump is treated as a constant-density liquid pump. The map reports
# head rise in meters. The pump component converts that head rise into pressure
# using density * g * head.

LOX = Propellant("lox", temperature=90)

g = 9.80665

LOXPumpDensity = State(LOX.density)
LOXPumpDesignVolumetricFlow = LOXPumpDesignMassFlow / LOX.density
LOXPumpDesignHead = (LOXPumpDesignDischargePressure - LOXPumpDesignInletPressure) / (LOX.density * g)

OxRichPreburnerTemperature = Preburner.temperature
OxRichTurbineDesignPressureRatio = OxRichPreburnerPressure / OxRichTurbineDownstreamPressure


# -----------------------------------------------------------------------------
# Size the turbine map from the LOX pump design power
# -----------------------------------------------------------------------------
# The turbine map should be sized so that the turbine produces roughly the same
# shaft power required by the pump at the design point. This makes the rotor
# torque balance meaningful instead of relying on an arbitrary turbine scaling.

LOXPumpDesignHydraulicPower = LOX.density * g * LOXPumpDesignHead * LOXPumpDesignVolumetricFlow
LOXPumpDesignShaftPower = LOXPumpDesignHydraulicPower / LOXPumpDesignEfficiency

OxRichDesignGasIn = CombustionGas(
    Preburner.gas_composition,
    basis="mass",
    pressure=OxRichPreburnerPressure,
    temperature=OxRichPreburnerTemperature,
)

OxRichDesignGasOutIdeal = CombustionGas(
    Preburner.gas_composition,
    basis="mass",
    pressure=OxRichTurbineDownstreamPressure,
    entropy=OxRichDesignGasIn.entropy,
)

OxRichTurbineDesignIdealTotalEnthalpyChange = OxRichDesignGasIn.enthalpy - OxRichDesignGasOutIdeal.enthalpy
OxRichTurbineDesignMassFlow = LOXPumpDesignShaftPower / (OxRichTurbineDesignEfficiency * OxRichTurbineDesignIdealTotalEnthalpyChange)


# -----------------------------------------------------------------------------
# Shared turbopump map file
# -----------------------------------------------------------------------------

turbopump_map_filename = "8turbopump"


'''
# -----------------------------------------------------------------------------
# Turbine map generation
# -----------------------------------------------------------------------------
# Run this section once to generate the top-level /turbine map in turbopump.h5.
# After the map has been generated, this section can be commented out.
#
# Inputs:
#   upstream_total_pressure       turbine inlet total pressure [Pa]
#   downstream_pressure           turbine discharge static/back pressure [Pa]
#   upstream_total_temperature    turbine inlet total temperature [K]
#   rotor_speed                   shaft speed [rpm]
#
# Outputs:
#   flow_parameter                mdot * sqrt(R*Tt) / Pt
#   torque                        turbine shaft torque [N*m]
#   ideal_total_enthalpy_change   ideal turbine total enthalpy drop [J/kg]
# -----------------------------------------------------------------------------

def turbine_map_point(
    upstream_total_pressure,
    downstream_pressure,
    upstream_total_temperature,
    rotor_speed,
):
    # Re-solve the preburner at the turbine inlet pressure. This lets the gas
    # composition vary with the preburner pressure used by the map point.
    Preburner.pressure = upstream_total_pressure
    turbine_gas_composition = Preburner.gas_composition

    pressure_ratio = upstream_total_pressure / downstream_pressure
    omega = (np.pi / 30.0) * rotor_speed

    # Turbine inlet gas state.
    gas_in = CombustionGas(
        turbine_gas_composition,
        basis="mass",
        pressure=upstream_total_pressure,
        temperature=upstream_total_temperature,
    )

    # Ideal isentropic turbine outlet state.
    gas_out_ideal = CombustionGas(
        turbine_gas_composition,
        basis="mass",
        pressure=downstream_pressure,
        entropy=gas_in.entropy,
    )

    ideal_total_enthalpy_change = gas_in.enthalpy - gas_out_ideal.enthalpy

    # Scale turbine mass flow around the design point. This simple map model is
    # not a real turbine design; it is only a smooth, physically reasonable map
    # for demonstrating the FullFlow coupling.
    mass_flow = OxRichTurbineDesignMassFlow
    mass_flow = mass_flow * (upstream_total_pressure / OxRichPreburnerPressure)
    mass_flow = mass_flow * np.sqrt(OxRichPreburnerTemperature / upstream_total_temperature)
    mass_flow = mass_flow * ((1.0 - 1.0 / pressure_ratio) / (1.0 - 1.0 / OxRichTurbineDesignPressureRatio))
    mass_flow = mass_flow * (OxRichTurbineRotorSpeed / rotor_speed) ** 0.05

    flow_parameter = mass_flow * np.sqrt(gas_in.gas_constant * upstream_total_temperature) / upstream_total_pressure
    shaft_power = OxRichTurbineDesignEfficiency * mass_flow * ideal_total_enthalpy_change
    torque = shaft_power / omega

    return {
        "flow_parameter": flow_parameter,
        "torque": torque,
        "ideal_total_enthalpy_change": ideal_total_enthalpy_change,
    }


generate_map(
    turbopump_map_filename,
    group="turbine",
    axes=[
        Axis.values(
            "upstream_total_pressure",
            [
                500.0 * bar,
                600.0 * bar,
                700.0 * bar,
            ],
            units="Pa",
        ),
        Axis.values(
            "downstream_pressure",
            [
                250.0 * bar,
                350.0 * bar,
                450.0 * bar,
            ],
            units="Pa",
        ),
        Axis.values(
            "upstream_total_temperature",
            [
                0.90 * OxRichPreburnerTemperature,
                1.00 * OxRichPreburnerTemperature,
                1.10 * OxRichPreburnerTemperature,
            ],
            units="K",
        ),
        Axis.values(
            "rotor_speed",
            [
                30000.0,
                36000.0,
                42000.0,
            ],
            units="rpm",
        ),
    ],
    evaluate=turbine_map_point,
    outputs=[
        "flow_parameter",
        "torque",
        "ideal_total_enthalpy_change",
    ],
    overwrite=True,
    raise_errors=True,
)

# Restore the preburner to the design pressure after map generation.
Preburner.pressure = OxRichPreburnerPressure


# -----------------------------------------------------------------------------
# LOX pump map generation
# -----------------------------------------------------------------------------
# Run this section once to generate the top-level /lox_pump map in turbopump.h5.
# After the map has been generated, this section can be commented out.
#
# Inputs:
#   rotor_speed        shaft speed [rpm]
#   volumetric_flow    LOX volumetric flow rate [m^3/s]
#
# Outputs:
#   head_rise          pump head rise [m]
#   torque             pump shaft torque [N*m]
# -----------------------------------------------------------------------------

def lox_pump_map_point(rotor_speed, volumetric_flow):
    omega = (np.pi / 30.0) * rotor_speed

    speed_ratio = rotor_speed / LOXPumpDesignRotorSpeed
    best_flow = LOXPumpDesignVolumetricFlow * speed_ratio

    # Simple pump curve:
    #   head scales with speed squared,
    #   best-flow point scales with speed,
    #   head falls off away from best flow.
    head_rise = LOXPumpDesignHead * speed_ratio**2
    head_rise = head_rise - 0.25 * LOXPumpDesignHead * ((volumetric_flow - best_flow) / LOXPumpDesignVolumetricFlow) ** 2
    head_rise = max(head_rise, 1.0)

    hydraulic_power = LOX.density * g * head_rise * volumetric_flow
    shaft_power = hydraulic_power / LOXPumpDesignEfficiency
    torque = shaft_power / omega

    return {
        "head_rise": head_rise,
        "torque": torque,
    }


generate_map(
    turbopump_map_filename,
    group="lox_pump",
    axes=[
        Axis.values(
            "rotor_speed",
            [
                30000.0,
                36000.0,
                42000.0,
            ],
            units="rpm",
        ),
        Axis.values(
            "volumetric_flow",
            [
                0.70 * LOXPumpDesignVolumetricFlow,
                1.00 * LOXPumpDesignVolumetricFlow,
                1.30 * LOXPumpDesignVolumetricFlow,
            ],
            units="m^3/s",
        ),
    ],
    evaluate=lox_pump_map_point,
    outputs=[
        "head_rise",
        "torque",
    ],
    overwrite=True,
    raise_errors=True,
)
'''


# -----------------------------------------------------------------------------
# Read the turbine map
# -----------------------------------------------------------------------------
# These State objects are the live inputs into the interpolation map. When the
# solver changes TurbineRotorSpeed, both the turbine map and LOX pump map see
# the same updated rotor speed.

TurbineUpstreamTotalPressure = State(Preburner.pressure)
TurbineUpstreamTotalTemperature = State(Preburner.temperature)
TurbineDownstreamPressure = State(OxRichTurbineDownstreamPressure)
TurbineRotorSpeed = State(OxRichTurbineRotorSpeed)

TurbineMap = Map.from_hdf5(
    "Turbine Map",
    Turbopump,
    filename=turbopump_map_filename,
    group="turbine",
    inputs={
        "upstream_total_pressure": TurbineUpstreamTotalPressure,
        "downstream_pressure": TurbineDownstreamPressure,
        "upstream_total_temperature": TurbineUpstreamTotalTemperature,
        "rotor_speed": TurbineRotorSpeed,
    },
    outputs=[
        "flow_parameter",
        "torque",
        "ideal_total_enthalpy_change",
    ],
)


# -----------------------------------------------------------------------------
# Read the LOX pump map
# -----------------------------------------------------------------------------
# The pump map uses the same rotor-speed State as the turbine map. The other
# input is volumetric flow rate, computed from mass flow and liquid density.

LOXPumpMassFlow = State(LOXPumpDesignMassFlow)
LOXPumpVolumetricFlow = LOXPumpMassFlow / LOXPumpDensity

LOXPumpMap = Map.from_hdf5(
    "LOX Pump Map",
    Turbopump,
    filename=turbopump_map_filename,
    group="lox_pump",
    inputs={
        "rotor_speed": TurbineRotorSpeed,
        "volumetric_flow": LOXPumpVolumetricFlow,
    },
    outputs=[
        "head_rise",
        "torque",
    ],
)


# -----------------------------------------------------------------------------
# LOX pump component
# -----------------------------------------------------------------------------
# ConstantDensityPump converts the mapped head rise into discharge pressure:
#
#   po_out = upstream_pressure + density * g * head_rise
#
# The pump mass flow is an iteration variable inside ConstantDensityPump, while
# the rotor speed is solved by the Rotor torque balance below.

LOXPumpInletPressure = State(LOXPumpDesignInletPressure)
LOXPumpDischargePressure = State(LOXPumpDesignDischargePressure)

LOXPump = ConstantDensityPump(
    "LOX Pump",
    Turbopump,
    mass_flow=LOXPumpMassFlow,
    rotor_speed=TurbineRotorSpeed,
    head_rise=LOXPumpMap.head_rise,
    density=LOXPumpDensity,
    torque=LOXPumpMap.torque,
    upstream_pressure=LOXPumpInletPressure,
    discharge_pressure=LOXPumpDischargePressure,
)


# -----------------------------------------------------------------------------
# Oxygen-rich turbine component
# -----------------------------------------------------------------------------
# GasTurbine converts the mapped flow parameter into mass flow and the mapped
# turbine torque into shaft power.

OxRichTurbine = GasTurbine(
    "Oxygen-Rich Turbine",
    Turbopump,
    rotor_speed=TurbineRotorSpeed,
    torque=TurbineMap.torque,
    flow_parameter=TurbineMap.flow_parameter,
    upstream_total_pressure=TurbineUpstreamTotalPressure,
    upstream_total_temperature=TurbineUpstreamTotalTemperature,
    downstream_pressure=TurbineDownstreamPressure,
    gas_constant=Preburner.gas_constant,
    specific_heat_ratio=Preburner.gamma,
    upstream_total_enthalpy=Preburner.enthalpy,
    ideal_total_enthalpy_change=TurbineMap.ideal_total_enthalpy_change,
)


# -----------------------------------------------------------------------------
# Rotor torque balance
# -----------------------------------------------------------------------------
# The rotor imposes steady-state torque balance on the shared shaft:
#
#   turbine torque - pump torque = 0
#
# The rotor speed is the iteration variable. The solver changes rotor speed
# until this residual is driven to zero.

TBRotorNetTorque = TurbineMap.torque - LOXPumpMap.torque

TBRotor = Rotor(
    "Turbopump Rotor",
    Turbopump,
    rotor_speed=TurbineRotorSpeed,
    polar_moment_of_inertia=1.0,
    net_torque=TBRotorNetTorque,
)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------

SteadyState(Turbopump).solve(verbose=True)
