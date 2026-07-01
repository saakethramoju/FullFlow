"""
Simple transient pump startup, valve closure, and pump shutdown example
with pump model options.

Physical layout
---------------

      Water source
      P = 100 psia
      T = 300 K
      Fluid lookup supplies density and viscosity
            |
            v
      Inlet Darcy-Weisbach line
      friction factor from Churchill correlation
            |
            v
      Pump inlet finite-volume node
      pressure and density from Fluid lookup
            |
            v
      Pump model
      Option 1: ConstantDensityPump
      Option 2: PolytropicPump
      speed ramps up, holds, then shuts down
            |
            v
      Pump discharge finite-volume node
      pressure and density from Fluid lookup
            |
            v
      Outlet valve
      starts open, then closes
            |
            v
      Outlet finite-volume node
      pressure and density from Fluid lookup
            |
            v
      Outlet Darcy-Weisbach line
      friction factor from Churchill correlation
            |
            v
      Drain / ambient receiver
      P = 100 psia
      T = 300 K
      Fluid lookup supplies density and viscosity


Model notes
-----------

This example is intentionally simple, but now uses model options.

The model named "Pump Model" has two options:

    1. Constant Density Pump
    2. Polytropic Pump

The pump head rise follows a simple speed-squared relation:

    head_rise = design_head_rise * (rotor_speed / design_rotor_speed)^2

The source and drain start at the same pressure, so the initial no-pump flow is
approximately zero.

The transient sequence is:

    1. Pump starts near 0 rpm with the outlet valve open.
    2. Pump spins up while the outlet valve is open.
    3. Pump holds at design speed.
    4. Outlet valve closes.
    5. Pump slowly shuts down.

The pump inlet, pump discharge, and outlet nodes are finite Volume components.
Their densities come from Fluid lookups, so each node stores mass:

    mass = density(P, T) * volume

and pressure evolves from mass conservation:

    d(mass)/dt = mdot_in - mdot_out

The transient solve evaluates every pump option independently.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

filename = "9model_options"


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

psi_to_pa = 6894.76
pa_to_psi = 1.0 / psi_to_pa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

g = 9.80665

source_pressure_value = 100.0 * psi_to_pa
drain_pressure_value = 100.0 * psi_to_pa

water_temperature_value = 300.0

design_rotor_speed = 6000.0

# The design pump pressure rise is about 150 psid.
water_density_reference = 997.0
design_pressure_rise = 150.0 * psi_to_pa
design_head_rise = design_pressure_rise / (water_density_reference * g)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

PumpNetwork = Network("Simple Transient Pump Model Options")


# ---------------------------------------------------------------------------
# Boundary states
# ---------------------------------------------------------------------------

source_pressure = State(source_pressure_value)
drain_pressure = State(drain_pressure_value)

water_temperature = State(water_temperature_value)


# ---------------------------------------------------------------------------
# Rotor-speed schedule
# ---------------------------------------------------------------------------

# This tiny idle speed behaves like zero for the example, but avoids the exact
# zero-pressure-ratio / zero-density-ratio singularity in PolytropicPump.
rotor_speed_idle = 1.0

rotor_speed = State(rotor_speed_idle)

rotor_start_delay = 0.5
rotor_ramp_time = 6.0
rotor_shutdown_delay = 12.0
rotor_shutdown_time = 8.0


def rotor_speed_schedule(t):
    """
    Spin the pump up, hold briefly, then slowly shut it down.

    A short start delay keeps the first few transient steps essentially at the
    initialized zero-flow condition.
    """

    if t <= rotor_start_delay:
        return rotor_speed_idle

    ramp_fraction = (t - rotor_start_delay) / rotor_ramp_time

    if ramp_fraction < 1.0:
        ramp_fraction = 0.5 - 0.5 * np.cos(np.pi * ramp_fraction)
        return rotor_speed_idle + (design_rotor_speed - rotor_speed_idle) * ramp_fraction

    if t < rotor_shutdown_delay:
        return design_rotor_speed

    ramp_fraction = (t - rotor_shutdown_delay) / rotor_shutdown_time

    if ramp_fraction >= 1.0:
        return rotor_speed_idle

    ramp_fraction = 0.5 - 0.5 * np.cos(np.pi * ramp_fraction)

    return rotor_speed_idle + (design_rotor_speed - rotor_speed_idle) * (1.0 - ramp_fraction)


RotorSpeedSequence = Sequence(
    "Rotor Speed",
    PumpNetwork,
    target=rotor_speed,
    function=rotor_speed_schedule,
)


# ---------------------------------------------------------------------------
# Outlet valve schedule
# ---------------------------------------------------------------------------

valve_cd_open = 0.75
valve_cd_closed = 1.0e-3

valve_cd = State(valve_cd_open)

valve_close_delay = 9.0
valve_close_time = 3.0


def valve_cd_schedule(t):
    """
    Keep the outlet valve open during pump spinup, then close it smoothly.

    A small nonzero closed Cd avoids a perfect shutoff singularity while still
    behaving like a nearly closed valve.
    """

    if t <= valve_close_delay:
        return valve_cd_open

    ramp_fraction = (t - valve_close_delay) / valve_close_time

    if ramp_fraction >= 1.0:
        return valve_cd_closed

    ramp_fraction = 0.5 - 0.5 * np.cos(np.pi * ramp_fraction)

    return valve_cd_open + (valve_cd_closed - valve_cd_open) * ramp_fraction


ValveSequence = Sequence(
    "Outlet Valve Cd",
    PumpNetwork,
    target=valve_cd,
    function=valve_cd_schedule,
)


# ---------------------------------------------------------------------------
# Pump head schedule
# ---------------------------------------------------------------------------

# Simple pump affinity-law behavior.
pump_head_rise = design_head_rise * (rotor_speed / design_rotor_speed) ** 2


# ---------------------------------------------------------------------------
# Flow and pressure states
# ---------------------------------------------------------------------------

inlet_line_mass_flow = State(0.0)
pump_mass_flow = State(0.0)
valve_mass_flow = State(0.0)
outlet_line_mass_flow = State(0.0)

pump_inlet_pressure = State(source_pressure_value)
pump_discharge_pressure = State(source_pressure_value + water_density_reference * g * pump_head_rise.value)
outlet_node_pressure = State(drain_pressure_value)


# ---------------------------------------------------------------------------
# Finite-volume node sizes
# ---------------------------------------------------------------------------

# These are effective hydraulic compliance volumes. They do not need to equal
# literal pipe volumes. Larger values make the liquid-pressure transient less
# stiff while still preserving finite-volume behavior.
pump_inlet_volume = 10.0
pump_discharge_volume = 30.0
outlet_node_volume = 10.0


# ---------------------------------------------------------------------------
# Line and valve geometry
# ---------------------------------------------------------------------------

inlet_line_length = 10.0
inlet_line_diameter = 0.040
inlet_line_area = (np.pi / 4.0) * inlet_line_diameter**2

outlet_line_length = 80.0
outlet_line_diameter = 0.030
outlet_line_area = (np.pi / 4.0) * outlet_line_diameter**2

valve_area = outlet_line_area

inlet_line_roughness = 1.0e-6
outlet_line_roughness = 1.0e-6

inlet_line_friction_factor = State(0.025)
outlet_line_friction_factor = State(0.025)

inlet_line_reynolds_number = State(1.0)
outlet_line_reynolds_number = State(1.0)


# ---------------------------------------------------------------------------
# Fluid property lookups
# ---------------------------------------------------------------------------

SourceWater = Lookup(
    "Source Water",
    PumpNetwork,
    Fluid,
    "Water",
    pressure=source_pressure,
    temperature=water_temperature,
)

PumpInletWater = Lookup(
    "Pump Inlet Water",
    PumpNetwork,
    Fluid,
    "Water",
    pressure=pump_inlet_pressure,
    temperature=water_temperature,
)

PumpDischargeWater = Lookup(
    "Pump Discharge Water",
    PumpNetwork,
    Fluid,
    "Water",
    pressure=pump_discharge_pressure,
    temperature=water_temperature,
)

OutletNodeWater = Lookup(
    "Outlet Node Water",
    PumpNetwork,
    Fluid,
    "Water",
    pressure=outlet_node_pressure,
    temperature=water_temperature,
)

DrainWater = Lookup(
    "Drain Water",
    PumpNetwork,
    Fluid,
    "Water",
    pressure=drain_pressure,
    temperature=water_temperature,
)


# ---------------------------------------------------------------------------
# Inlet line friction factor
# ---------------------------------------------------------------------------

InletLineFriction = Churchill(
    "Inlet Line Friction",
    PumpNetwork,
    mass_flow=inlet_line_mass_flow,
    friction_factor=inlet_line_friction_factor,
    hydraulic_diameter=inlet_line_diameter,
    dynamic_viscosity=SourceWater.dynamic_viscosity,
    cross_sectional_area=inlet_line_area,
    roughness=inlet_line_roughness,
    reynolds_number=inlet_line_reynolds_number,
)


# ---------------------------------------------------------------------------
# Inlet Darcy-Weisbach line
# ---------------------------------------------------------------------------

InletLine = DarcyWeisbach(
    "Inlet Line",
    PumpNetwork,
    mass_flow=inlet_line_mass_flow,
    upstream_pressure=source_pressure,
    downstream_pressure=pump_inlet_pressure,
    length=inlet_line_length,
    hydraulic_diameter=inlet_line_diameter,
    cross_sectional_area=inlet_line_area,
    density=SourceWater.density,
    friction_factor=inlet_line_friction_factor,
)


# ---------------------------------------------------------------------------
# Pump inlet finite-volume node
# ---------------------------------------------------------------------------

PumpInletNode = Volume(
    "Pump Inlet Node",
    PumpNetwork,
    pressure=pump_inlet_pressure,
    volume=pump_inlet_volume,
    density=PumpInletWater.density,
    mass_flow_in=InletLine.mass_flow,
    mass_flow_out=pump_mass_flow,
)


# ---------------------------------------------------------------------------
# Pump model options
# ---------------------------------------------------------------------------

PumpModel = Model("Pump Model", PumpNetwork)

PumpModel.option(
    "Constant Density Pump",
    ConstantDensityPump.template(
        "Pump",
        mass_flow=pump_mass_flow,
        rotor_speed=rotor_speed,
        head_rise=pump_head_rise,
        density=PumpInletWater.density,
        upstream_pressure=PumpInletNode.pressure,
        discharge_pressure=pump_discharge_pressure,
    ),
)

PumpModel.option(
    "Polytropic Pump",
    PolytropicPump.template(
        "Pump",
        mass_flow=pump_mass_flow,
        rotor_speed=rotor_speed,
        head_rise=pump_head_rise,
        upstream_pressure=PumpInletNode.pressure,
        discharge_pressure=pump_discharge_pressure,
        upstream_density=PumpInletWater.density,
        discharge_density=PumpDischargeWater.density,
    ),
)


# ---------------------------------------------------------------------------
# Pump discharge finite-volume node
# ---------------------------------------------------------------------------

PumpDischargeNode = Volume(
    "Pump Discharge Node",
    PumpNetwork,
    pressure=pump_discharge_pressure,
    volume=pump_discharge_volume,
    density=PumpDischargeWater.density,
    mass_flow_in=pump_mass_flow,
    mass_flow_out=valve_mass_flow,
)


# ---------------------------------------------------------------------------
# Outlet valve
# ---------------------------------------------------------------------------

OutletValve = DischargeCoefficient(
    "Outlet Valve",
    PumpNetwork,
    upstream_pressure=PumpDischargeNode.pressure,
    downstream_pressure=outlet_node_pressure,
    density=PumpDischargeWater.density,
    discharge_coefficient=valve_cd,
    cross_sectional_area=valve_area,
    mass_flow=valve_mass_flow,
)


# ---------------------------------------------------------------------------
# Outlet finite-volume node
# ---------------------------------------------------------------------------

OutletNode = Volume(
    "Outlet Node",
    PumpNetwork,
    pressure=outlet_node_pressure,
    volume=outlet_node_volume,
    density=OutletNodeWater.density,
    mass_flow_in=OutletValve.mass_flow,
    mass_flow_out=outlet_line_mass_flow,
)


# ---------------------------------------------------------------------------
# Outlet line friction factor
# ---------------------------------------------------------------------------

OutletLineFriction = Churchill(
    "Outlet Line Friction",
    PumpNetwork,
    mass_flow=outlet_line_mass_flow,
    friction_factor=outlet_line_friction_factor,
    hydraulic_diameter=outlet_line_diameter,
    dynamic_viscosity=OutletNodeWater.dynamic_viscosity,
    cross_sectional_area=outlet_line_area,
    roughness=outlet_line_roughness,
    reynolds_number=outlet_line_reynolds_number,
)


# ---------------------------------------------------------------------------
# Outlet Darcy-Weisbach line
# ---------------------------------------------------------------------------

OutletLine = DarcyWeisbach(
    "Outlet Line",
    PumpNetwork,
    mass_flow=outlet_line_mass_flow,
    upstream_pressure=OutletNode.pressure,
    downstream_pressure=drain_pressure,
    length=outlet_line_length,
    hydraulic_diameter=outlet_line_diameter,
    cross_sectional_area=outlet_line_area,
    density=OutletNodeWater.density,
    friction_factor=outlet_line_friction_factor,
)


# ---------------------------------------------------------------------------
# Tracked outputs
# ---------------------------------------------------------------------------

PumpNetwork.track("Rotor Speed [rpm]", rotor_speed)
PumpNetwork.track("Outlet Valve Cd [-]", valve_cd)

PumpNetwork.track("Inlet Line Mass Flow [kg/s]", inlet_line_mass_flow)
PumpNetwork.track("Pump Mass Flow [kg/s]", pump_mass_flow)
PumpNetwork.track("Valve Mass Flow [kg/s]", valve_mass_flow)
PumpNetwork.track("Outlet Line Mass Flow [kg/s]", outlet_line_mass_flow)

PumpNetwork.track("Source Pressure [psia]", source_pressure * pa_to_psi)
PumpNetwork.track("Pump Inlet Pressure [psia]", pump_inlet_pressure * pa_to_psi)
PumpNetwork.track("Pump Discharge Pressure [psia]", pump_discharge_pressure * pa_to_psi)
PumpNetwork.track("Outlet Node Pressure [psia]", outlet_node_pressure * pa_to_psi)
PumpNetwork.track("Drain Pressure [psia]", drain_pressure * pa_to_psi)

PumpNetwork.track("Pump Pressure Rise [psid]", (pump_discharge_pressure - pump_inlet_pressure) * pa_to_psi)
PumpNetwork.track("Pump Head Rise [m]", pump_head_rise)

PumpNetwork.track("Source Density [kg/m^3]", SourceWater.density)
PumpNetwork.track("Source Dynamic Viscosity [Pa*s]", SourceWater.dynamic_viscosity)

PumpNetwork.track("Pump Inlet Density [kg/m^3]", PumpInletWater.density)
PumpNetwork.track("Pump Discharge Density [kg/m^3]", PumpDischargeWater.density)
PumpNetwork.track("Outlet Node Density [kg/m^3]", OutletNodeWater.density)
PumpNetwork.track("Drain Density [kg/m^3]", DrainWater.density)

PumpNetwork.track("Inlet Line Friction Factor [-]", inlet_line_friction_factor)
PumpNetwork.track("Outlet Line Friction Factor [-]", outlet_line_friction_factor)

PumpNetwork.track("Inlet Line Reynolds Number [-]", inlet_line_reynolds_number)
PumpNetwork.track("Outlet Line Reynolds Number [-]", outlet_line_reynolds_number)

PumpNetwork.track("Pump Inlet Stored Mass [kg]", PumpInletNode.mass)
PumpNetwork.track("Pump Discharge Stored Mass [kg]", PumpDischargeNode.mass)
PumpNetwork.track("Outlet Node Stored Mass [kg]", OutletNode.mass)


# ---------------------------------------------------------------------------
# Steady-state initialization
# ---------------------------------------------------------------------------

# Initialize using the first successful option in Pump Model.
SteadyState(PumpNetwork).solve(
    verbose=True,
    filename=filename,
    model="Pump Model",
)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------

# Evaluate both pump options as independent transient runs.
Transient(PumpNetwork).solve(
    dt=0.01,
    t_final=22.0,
    filename=filename,
    model="Pump Model",
    evaluate_all_model_options=True,
    statistics=True,
    state_max_passes=30,
)