"""
Transient pump / valve / pipeline startup example.

Physical layout
---------------

    Supply reservoir
    P = 14.7 psia
    T = 60 F
          |
          v
      +--------+
      |  Pump  |  head-rise map:
      +--------+      H = f(Q)
          |
          v
    Pump outlet node
          |
          v
      Gate valve
      nearly closed for steady initialization,
      then smoothly ramps open during transient
          |
          v
    Pipe inlet node
          |
          v
    1500 ft pipe, 6 in diameter
    receiving reservoir is 150 ft above supply reservoir
          |
          v
    Receiving reservoir
    P = 14.7 psia
    T = 60 F


Model notes
-----------

The pump map is generated into the same HDF5 file used for the steady-state
and transient results.

The pump map uses volumetric flow rate as the independent variable and returns:

    head_rise
    pressure_rise

The pump component is used without torque. Because torque is not assigned,
rotor_speed is arbitrary and no torque-dependent outputs are calculated by
ConstantDensityPump. The pump simply uses the mapped head rise to calculate
the discharge pressure rise.

The initial steady-state solve is performed with the valve nearly closed. This
starts the transient from a dead-head-like pump condition. During the transient,
the valve opens smoothly, the flow rate increases, the pump head decreases, and
the pump operating point moves along the head-flow curve.

The gate valve is modeled as an algebraic discharge-coefficient branch. The
long downstream pipe still carries the main flow inertia, so the transient
shows the system response without making the tiny closed-valve Cd create a
stiff inertial valve equation.

This example is loosely based on a GFSSP pump / valve / pipeline example. It is
not intended to be a one-to-one reproduction of the GFSSP model.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

filename = "pump_transient_example"


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

psi_to_pa = 6894.76

ft_to_m = 0.3048
inch_to_m = 0.0254

lbm_to_kg = 0.45359237
kg_s_to_lbm_s = 1.0 / lbm_to_kg

gallon_to_m3 = 0.003785411784
minute_to_second = 60.0

gpm_to_m3_s = gallon_to_m3 / minute_to_second
m3_s_to_gpm = 1.0 / gpm_to_m3_s


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

g = 9.80665

water_density_value = 62.4 * lbm_to_kg / ft_to_m**3
water_dynamic_viscosity_value = 1.12e-3

source_pressure_value = 14.7 * psi_to_pa
receiver_pressure_value = 14.7 * psi_to_pa

pipe_length = 1500 * ft_to_m
pipe_diameter = 6 * inch_to_m
pipe_area = (np.pi / 4) * pipe_diameter**2

pipe_relative_roughness = 0.005
pipe_roughness = pipe_relative_roughness * pipe_diameter

receiver_height_change = 150 * ft_to_m

# Rotor speed is arbitrary in this example because torque is not assigned to
# ConstantDensityPump. The pump only uses head_rise to calculate pressure rise.
rotor_speed_value = 1.0


# ---------------------------------------------------------------------------
# Generate pump head-flow map
# ---------------------------------------------------------------------------

# Manufacturer-style pump curve points.
#
# The independent variable is volumetric flow rate in gallons per minute.
# The dependent variable is total pump head in feet.
pump_flow_rate_points = np.array([
    0.0,
    4000.0,
    8000.0,
    12000.0,
    16000.0,
    20000.0,
])

pump_head_rise_points = np.array([
    495.0,
    485.0,
    470.0,
    450.0,
    425.0,
    385.0,
])

# A simple quadratic is used only to make a smooth pump map from the tabulated
# head-flow points above.
pump_curve_coefficients = np.polyfit(
    pump_flow_rate_points,
    pump_head_rise_points,
    2,
)


def pump_curve(flow_rate):
    """
    Evaluate the pump head-flow curve.

    Parameters
    ----------
    flow_rate : float
        Volumetric flow rate in gallons per minute.

    Returns
    -------
    dict
        Pump head rise and pressure rise.
    """

    head_rise_ft = np.polyval(pump_curve_coefficients, flow_rate)

    head_rise = head_rise_ft * ft_to_m
    pressure_rise = water_density_value * g * head_rise

    return {
        "head_rise": head_rise,
        "pressure_rise": pressure_rise,
    }


# The pump map is stored in the same HDF5 file that will later hold the steady
# and transient results.
generate_map(
    filename,
    group="pump_curve",
    axes=[
        Axis.linear(
            "flow_rate",
            start=0.0,
            stop=20000.0,
            count=101,
            units="gpm",
        ),
    ],
    evaluate=pump_curve,
    outputs=[
        "head_rise",
        "pressure_rise",
    ],
    metadata={
        "flow_rate_units": "gpm",
        "head_rise_units": "m",
        "pressure_rise_units": "Pa",
    },
    overwrite=True,
)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

PumpSystem = Network("Pump Transient System")


# ---------------------------------------------------------------------------
# Fluid and boundary states
# ---------------------------------------------------------------------------

water_density = State(water_density_value)
water_dynamic_viscosity = State(water_dynamic_viscosity_value)

source_pressure = State(source_pressure_value)
receiver_pressure = State(receiver_pressure_value)

rotor_speed = State(rotor_speed_value)


# ---------------------------------------------------------------------------
# Valve states
# ---------------------------------------------------------------------------

# The valve starts nearly closed. A small nonzero Cd avoids a perfectly closed
# algebraic singularity while still creating a dead-head-like pump condition.
#
# A value of 1.0e-3 is intentionally less stiff than 1.0e-4, because
# discharge-coefficient resistance scales approximately with 1 / Cd^2.
valve_cd_closed = 1.0e-3
valve_cd_open = 0.75

valve_cd = State(valve_cd_closed)


# ---------------------------------------------------------------------------
# Flow and pressure states
# ---------------------------------------------------------------------------

# Start near zero flow for the closed-valve steady solve.
pump_mass_flow = State(0.01)
valve_mass_flow = State(0.01)
pipe_mass_flow = State(0.01)

# Initial pressure guesses.
#
# The pump discharge guess is near shutoff pressure. The pipe inlet guess is
# near the receiver pressure plus static elevation head.
pump_discharge_pressure = State(source_pressure_value + water_density_value * g * 495.0 * ft_to_m)
pipe_inlet_pressure = State(receiver_pressure_value + water_density_value * g * receiver_height_change)

pipe_friction_factor = State(0.03)


# ---------------------------------------------------------------------------
# Pump curve map read from the same HDF5 file as the results
# ---------------------------------------------------------------------------

# Convert the pump mass flow to volumetric flow in gallons per minute for the
# pump map input.
pump_flow_rate = pump_mass_flow / water_density * m3_s_to_gpm

PumpCurve = Map.from_hdf5(
    "Pump Curve",
    PumpSystem,
    filename=filename,
    group="pump_curve",
    inputs={
        "flow_rate": pump_flow_rate,
    },
    outputs=[
        "head_rise",
        "pressure_rise",
    ],
    extrapolate=True,
)


# ---------------------------------------------------------------------------
# Pump
# ---------------------------------------------------------------------------

# Torque is intentionally omitted. ConstantDensityPump uses the mapped head rise
# to calculate discharge pressure. Shaft speed is arbitrary because no torque-
# dependent outputs are being calculated.
Pump = ConstantDensityPump(
    "Pump",
    PumpSystem,
    mass_flow=pump_mass_flow,
    rotor_speed=rotor_speed,
    head_rise=PumpCurve.head_rise,
    density=water_density,
    upstream_pressure=source_pressure,
    discharge_pressure=pump_discharge_pressure,
)


# ---------------------------------------------------------------------------
# Pump outlet / valve inlet node
# ---------------------------------------------------------------------------

# This node enforces continuity between the pump and the gate valve.
PumpOutlet = Volume(
    "Pump Outlet",
    PumpSystem,
    pressure=pump_discharge_pressure,
    density=water_density,
    mass_flow_in=Pump.mass_flow,
    mass_flow_out=valve_mass_flow,
)


# ---------------------------------------------------------------------------
# Gate valve opening schedule
# ---------------------------------------------------------------------------

valve_open_delay = 0.5
valve_open_time = 3.0


def valve_opening(t):
    """
    Smoothly ramp the valve from nearly closed to open.

    The steady-state solve sees the initial closed value because the Sequence
    only actively updates during the transient solve.

    A cosine smoothstep is used instead of a linear ramp. This avoids a sharp
    derivative change at the beginning of valve motion and reduces transient
    step retries.
    """

    if t <= valve_open_delay:
        return valve_cd_closed

    ramp_fraction = (t - valve_open_delay) / valve_open_time

    if ramp_fraction >= 1.0:
        return valve_cd_open

    ramp_fraction = 0.5 - 0.5 * np.cos(np.pi * ramp_fraction)

    return valve_cd_closed + (valve_cd_open - valve_cd_closed) * ramp_fraction


ValveSequence = Sequence(
    "Valve Cd",
    PumpSystem,
    target=valve_cd,
    function=valve_opening,
)


# ---------------------------------------------------------------------------
# Gate valve
# ---------------------------------------------------------------------------

# The valve is modeled as an algebraic discharge-coefficient branch.
#
# Length is intentionally not supplied here. If length is supplied, the valve
# becomes an inertial branch. With a very small closed Cd, that creates a stiff
# dynamic equation at the start of opening because valve resistance scales like
# 1 / Cd^2.
GateValve = DischargeCoefficient(
    "Gate Valve",
    PumpSystem,
    upstream_pressure=PumpOutlet.pressure,
    downstream_pressure=pipe_inlet_pressure,
    density=water_density,
    discharge_coefficient=valve_cd,
    cross_sectional_area=pipe_area,
    mass_flow=valve_mass_flow,
)


# ---------------------------------------------------------------------------
# Valve outlet / pipe inlet node
# ---------------------------------------------------------------------------

# This node enforces continuity between the valve and the long downstream pipe.
PipeInlet = Volume(
    "Pipe Inlet",
    PumpSystem,
    pressure=pipe_inlet_pressure,
    density=water_density,
    mass_flow_in=GateValve.mass_flow,
    mass_flow_out=pipe_mass_flow,
)


# ---------------------------------------------------------------------------
# Pipe friction factor
# ---------------------------------------------------------------------------

# The friction factor is solved from the current pipe flow rate.
PipeFriction = Colebrook(
    "Pipe Friction Factor",
    PumpSystem,
    mass_flow=pipe_mass_flow,
    friction_factor=pipe_friction_factor,
    hydraulic_diameter=pipe_diameter,
    dynamic_viscosity=water_dynamic_viscosity,
    cross_sectional_area=pipe_area,
    roughness=pipe_roughness,
)


# ---------------------------------------------------------------------------
# Long pipe to receiving reservoir
# ---------------------------------------------------------------------------

# The long pipe contains the main flow inertia and friction loss for the
# transient. As the valve opens, pipe_mass_flow evolves toward the open-valve
# operating point.
Pipe = DarcyWeisbach(
    "Pipe",
    PumpSystem,
    mass_flow=pipe_mass_flow,
    upstream_pressure=PipeInlet.pressure,
    downstream_pressure=receiver_pressure,
    length=pipe_length,
    hydraulic_diameter=pipe_diameter,
    cross_sectional_area=pipe_area,
    density=water_density,
    friction_factor=pipe_friction_factor,
    height_change=receiver_height_change,
)


# ---------------------------------------------------------------------------
# Tracked outputs
# ---------------------------------------------------------------------------

PumpSystem.track("Valve Cd [-]", valve_cd)

PumpSystem.track("Pump Flow Rate [gpm]", pump_flow_rate)
PumpSystem.track("Pump Mass Flow [lbm/s]", pump_mass_flow * kg_s_to_lbm_s)
PumpSystem.track("Valve Mass Flow [lbm/s]", valve_mass_flow * kg_s_to_lbm_s)
PumpSystem.track("Pipe Mass Flow [lbm/s]", pipe_mass_flow * kg_s_to_lbm_s)

PumpSystem.track("Pump Head Rise [ft]", PumpCurve.head_rise / ft_to_m)
PumpSystem.track("Pump Pressure Rise [psi]", PumpCurve.pressure_rise / psi_to_pa)

PumpSystem.track("Pump Discharge Pressure [psia]", pump_discharge_pressure / psi_to_pa)
PumpSystem.track("Pipe Inlet Pressure [psia]", pipe_inlet_pressure / psi_to_pa)

PumpSystem.track("Pipe Friction Factor [-]", pipe_friction_factor)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

# First solve the closed-valve operating point. This places the pump near
# shutoff head with very small flow.
SteadyState(PumpSystem).solve(
    verbose=True,
    filename=filename,
)

# Then open the valve during transient. Flow increases, pump head decreases,
# and the discharge pressure evolves as the system moves to the open-valve
# operating point.
Transient(PumpSystem).solve(
    dt=0.01,
    t_final=6.0,
    filename=filename,
    statistics=True,
)
