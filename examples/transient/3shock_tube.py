"""
Transient FlowTube shock-diagnostic decay test.

Physical layout
---------------

    Upstream Gas Node                         Downstream Gas Node
    P1, T1, rho1, a1                          P2, T2, rho2, a2
          |                                          |
          v                                          v

    +------------------------------------------------------+
    |              constant-area inviscid FlowTube          |
    |                                                      |
    |              no friction                             |
    |              no gravity                              |
    |              transient mass-flow state                |
    |                                                      |
    |              shock diagnostic only                    |
    +------------------------------------------------------+

Purpose
-------
This example tests the FlowTube normal-shock diagnostic during a transient.

The upstream and downstream gas states are fixed to the exact ideal-gas
normal-shock states for an upstream shock Mach number of M1 = 2.0.

The FlowTube mass flow is initialized below the exact stationary-shock mass
flow. Therefore, the mass flow changes during the transient, and the predicted
shock Mach number changes with it.

Important modeling note
-----------------------
The FlowTube does not solve a moving-shock problem here. It only checks whether
the current endpoint states and current mass flow look like a normal shock.

The FlowTube shock diagnostic should report:

    normal_shock      -> True while the current mass flow is shock-like
    shock_mach_number -> current upstream Mach number into the shock

Because the endpoint states are fixed, the changing shock Mach number comes
from the changing FlowTube mass flow.
"""
import math
from fullflow import *


ShockTube = Network("FlowTube Shock Decay")


# ---------------------------------------------------------------------------
# Ideal-gas lookup function
# ---------------------------------------------------------------------------
# The Lookup takes pressure and temperature as inputs, then returns:
#
#   density
#   speed of sound
#   static enthalpy
#
# These are the only gas properties needed by the FlowTube for this example.
# The tuple order must match the `outputs=(...)` names used in the Lookup.
# ---------------------------------------------------------------------------

def ideal_air_tuple(pressure, temperature):
    gamma = 1.4
    gas_constant = 287.0

    rho = pressure / (gas_constant * temperature)
    a = math.sqrt(gamma * gas_constant * temperature)
    h = gamma * gas_constant / (gamma - 1.0) * temperature

    return rho, a, h


# ---------------------------------------------------------------------------
# Build exact ideal-gas normal-shock endpoint states
# ---------------------------------------------------------------------------
# These equations create the downstream state that corresponds to an ideal-gas
# normal shock with upstream Mach number M1 = 2.0.
#
# The FlowTube will then be given those endpoint states and allowed to evolve
# its mass flow dynamically.
# ---------------------------------------------------------------------------

gamma = 1.4
gas_constant = 287.0

design_shock_mach_number = 2.0

P1 = 100_000.0      # upstream static pressure [Pa]
T1 = 300.0          # upstream static temperature [K]

P2_P1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (design_shock_mach_number**2 - 1.0)

rho2_rho1 = (
    (gamma + 1.0) * design_shock_mach_number**2
    / ((gamma - 1.0) * design_shock_mach_number**2 + 2.0)
)

T2_T1 = P2_P1 / rho2_rho1

P2 = P1 * P2_P1
T2 = T1 * T2_T1


# ---------------------------------------------------------------------------
# Lookup input states
# ---------------------------------------------------------------------------
# These are fixed boundary states for this simple diagnostic example.
# In a larger model, these could be states solved by volumes, chambers,
# tanks, or other connected components.
# ---------------------------------------------------------------------------

UpstreamPressure = State(P1)
UpstreamTemperature = State(T1)

DownstreamPressure = State(P2)
DownstreamTemperature = State(T2)


# ---------------------------------------------------------------------------
# Ideal-gas lookup nodes
# ---------------------------------------------------------------------------
# The Lookup outputs are exposed as:
#
#   UpstreamGas.density
#   UpstreamGas.speed_of_sound
#   UpstreamGas.static_enthalpy
#
# and similarly for DownstreamGas.
# ---------------------------------------------------------------------------

UpstreamGas = Lookup(
    "Upstream Gas",
    ShockTube,
    ideal_air_tuple,
    pressure=UpstreamPressure,
    temperature=UpstreamTemperature,
    outputs=("density", "speed_of_sound", "static_enthalpy"),
)

DownstreamGas = Lookup(
    "Downstream Gas",
    ShockTube,
    ideal_air_tuple,
    pressure=DownstreamPressure,
    temperature=DownstreamTemperature,
    outputs=("density", "speed_of_sound", "static_enthalpy"),
)


# ---------------------------------------------------------------------------
# FlowTube geometry
# ---------------------------------------------------------------------------

tube_diameter = 0.1                                      # [m]
tube_area = math.pi / 4.0 * tube_diameter**2             # [m^2]

# A longer tube makes the mass-flow transient slower and easier to visualize.
tube_length = 100.0                                      # [m]


# ---------------------------------------------------------------------------
# Initial mass flow
# ---------------------------------------------------------------------------
# The exact stationary-shock mass flow is:
#
#   mdot = rho1 * M1 * a1 * A
#
# Starting below this value makes the mass flow and shock Mach number change
# during the transient.
# ---------------------------------------------------------------------------

rho1 = P1 / (gas_constant * T1)
a1 = math.sqrt(gamma * gas_constant * T1)

exact_shock_mass_flow = rho1 * design_shock_mach_number * a1 * tube_area

initial_mass_flow = 0.95 * exact_shock_mass_flow

MassFlow = State(initial_mass_flow)


# ---------------------------------------------------------------------------
# Constant-area inviscid FlowTube
# ---------------------------------------------------------------------------
# No friction factor and no height change are provided. Therefore, the shock
# diagnostic is allowed to activate.
#
# The normal_shock and shock_mach_number outputs are diagnostic states. They do
# not change the FlowTube momentum equation.
# ---------------------------------------------------------------------------

Tube = FlowTube(
    "Shock Tube",
    ShockTube,
    mass_flow=MassFlow,
    upstream_static_pressure=UpstreamPressure,
    downstream_static_pressure=DownstreamPressure,
    length=tube_length,
    hydraulic_diameter=tube_diameter,
    cross_sectional_area=tube_area,
    upstream_density=UpstreamGas.density,
    downstream_density=DownstreamGas.density,
    upstream_speed_of_sound=UpstreamGas.speed_of_sound,
    downstream_speed_of_sound=DownstreamGas.speed_of_sound,
    upstream_static_enthalpy=UpstreamGas.static_enthalpy,
    normal_shock=False,
    shock_mach_number=0.0,
)


# ---------------------------------------------------------------------------
# Track outputs
# ---------------------------------------------------------------------------

ShockTube.track("Mass Flow [kg/s]", Tube.mass_flow)
ShockTube.track("Normal Shock", Tube.normal_shock)
ShockTube.track("Shock Mach Number", Tube.shock_mach_number)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------

filename = "flow_tube_shock_decay"

Transient(ShockTube).solve(
    t_final=0.30,
    dt=0.001,
    verbose=True,
    statistics=True,
    filename=filename,
)
