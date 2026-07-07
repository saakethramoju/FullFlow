import numpy as np

from fullflow import *
from thermoprop import *
import fullplot as fplt


"""
Dual-Propellant Dry-Line Priming and Ignition Startup Example
============================================================

Purpose
-------
This example demonstrates a reduced-order engine startup model in FullFlow.
It is intended to show how a transient simulation can be built like a small
virtual test stand:

    - the main valves are driven by command traces;
    - sensors watch simulated instrumentation values;
    - bluelines define ignition-permissive pressure thresholds;
    - a latched ignition command is generated from those sensor events;
    - the injector/chamber/nozzle system then evolves from the solved physics.

The case represents two initially air-filled propellant paths:

    - RP-1 fuel side
    - LOX oxidizer side

The startup sequence is intentionally split into two phases:

    1. Unlit priming initialization
       The feed system is solved before chamber combustion is added. This gives
       the dry-line model a clean pre-start state with the main valves closed,
       the injector outlets at ambient pressure, and combustion progress equal
       to zero.

    2. Transient startup with ignition
       During the transient solve, the main valves open, liquid fills the lines,
       the injector pressure bluelines latch, and the ignition command turns on.
       The ignition command does not directly force chamber pressure or nozzle
       flow. Instead, it drives a physical combustion-progress state whose time
       constant is the chamber residence time.

Physical layout
---------------
Each propellant side is modeled as:

    source liquid
        -> scheduled main valve CdA
        -> valve outlet junction
        -> variable-length wet liquid line
        -> priming liquid volume / injector pressure node
        -> shared downstream air volume
        -> open injector orifices
        -> chamber pressure boundary

The injector holes are always open. They are not scheduled valves. Before liquid
breakthrough, the holes are mostly exposed to air. After breakthrough, the same
physical injector CdA is gradually transferred from the air path to the liquid
path using an injector liquid-exposure fraction.

Chamber and combustion model
----------------------------
After the unlit steady-state initialization, a chamber gas map, isentropic nozzle,
and combustion chamber volume are added. The chamber pressure is not commanded.
It is the pressure state of a chamber Volume whose mass balance is:

    d(m_chamber)/dt = reacted_propellant_mass_flow - nozzle_mass_flow

The model separates injected propellant into two reduced-order streams:

    reacted_propellant_mass_flow   = combustion_progress * injector_total_mass_flow
    unreacted_propellant_mass_flow = (1 - combustion_progress) * injector_total_mass_flow

Only the reacted portion enters the hot-gas chamber volume. The unreacted portion
is tracked as an effective drain/bypass path. This is a reduced-order version of
"unburn" behavior, not a finite-rate chemistry model.

Combustion progress is modeled as:

    d(progress)/dt = (ignition_command - progress) / residence_time

where:

    residence_time = chamber_gas_inventory / injector_total_mass_flow

This makes the ignition transition depend on chamber gas inventory and injector
mass flow instead of an arbitrary command ramp. The ignition command is still a
sequence event, but the burn-in is a dynamic state.

What this example is and is not
------------------------------
This example is physical in the reduced-order system-modeling sense:

    - main valve motion is commanded;
    - line priming depends on volume filling and pressure balance;
    - ignition is enabled by simulated pressure sensors crossing bluelines;
    - chamber pressure comes from mass storage and nozzle outflow;
    - nozzle flow comes from the nozzle component;
    - gas properties come from the RP-1/LOX equilibrium map.

It is not a detailed ignition or combustion-stability model. It does not resolve
spray atomization, droplet evaporation, flame spreading, finite-rate chemistry,
local recirculation, or chamber acoustics. The burned/unburned split is a
one-state residence-time approximation suitable for a lumped transient example.

Files
-----
The combustion map is stored in the same HDF5 file used for results. Set
``generate_combustion_map = True`` once to generate the map, then set it back to
``False`` for normal runs.
"""
# Output HDF5 file stem. FullFlow and FullPlot will use this file for both
# generated map data and transient results.
filename = 'test'

# Set this to True only when the RP-1/LOX product-property map needs to be
# regenerated. Normal example runs should leave it False so startup simulation
# does not spend time rebuilding the map.
generate_combustion_map = False

mixture_ratio_map_min = 0.5
mixture_ratio_map_max = 6.0

# -----------------------------------------------------------------------------
# Constants and example parameters
# -----------------------------------------------------------------------------
# This section is deliberately kept near the top so a user can tune the example
# without digging through the model definition. The values are not intended to
# represent a specific engine; they produce a compact startup transient with
# visible priming, ignition, chamber-pressure rise, and nozzle-flow response.

# Unit conversions used only to keep the example readable.
psia_to_pa = 6894.76
in_to_m = 1.0 / 39.37
in3_to_m3 = in_to_m**3

# Transient integration settings. The timestep is small enough to resolve the
# valve opening, line filling, and ignition response in this example.
dt = 0.001
t_final = 0.35

# Ambient pressure is also the initial chamber and injector outlet pressure.
ambient_pressure = 14.67 * psia_to_pa

# Simple fixed source pressures. The tanks/regulators are not modeled here;
# they are treated as ideal upstream pressure boundaries.
fuel_source_pressure = 450.0 * psia_to_pa
ox_source_pressure = 400.0 * psia_to_pa

# Propellant and trapped-air temperatures. The line priming model is isothermal
# on each side; the chamber gas temperature comes from the combustion map.
fuel_temperature = 300.0
ox_temperature = 90.0
air_temperature = 300.0

# Downstream line geometry. The wet liquid line grows from the initial wet
# length to this full length as liquid displaces air.
line_length = 0.30
line_area = 0.5e-4
line_diameter = np.sqrt(4.0 * line_area / np.pi)
line_volume = line_area * line_length

# Manifold volume is lumped together with the line volume for the liquid/air
# fill-fraction calculation.
manifold_volume = 20.0 * in3_to_m3
total_downstream_volume = line_volume + manifold_volume

# Fully open main-valve effective areas. The command traces ramp these from
# zero to the open values.
fuel_main_cda_open = 0.5e-4
ox_main_cda_open = 1.0e-4

# Physical injector effective areas. Each area is split between air-exposed and
# liquid-exposed paths, but the two pieces always sum to the physical CdA.
fuel_injector_cda = 0.5e-4
ox_injector_cda = 1.0e-4

# A small initial wet length avoids a zero-length liquid line at t=0 and gives
# the solver a well-defined starting state.
initial_wet_length = 0.02
initial_liquid_volume = initial_wet_length * line_area

# Small smoothing length used by smooth_min/smooth_max. This prevents sharp
# kinks in algebraic expressions such as wet line length and clipped fractions.
wet_length_smoothing = 1.0e-5

# Main-valve schedule. Each main valve starts opening at 0.05 s and reaches the
# commanded open CdA after another 0.05 s.
main_valve_start_time = 0.05
main_valve_open_time = 0.05

# Injector wetting is modeled as a smooth exposure transition. When the total
# downstream fill fraction is below injector_wet_start, the injector holes are
# effectively air-exposed. When it reaches injector_wet_end, they are effectively
# liquid-exposed.
injector_wet_start = 0.90
injector_wet_end = 0.99

# Chamber/nozzle geometry used after the unlit feed-system initialization.
# Chamber pressure is solved by the chamber Volume; it is not prescribed.
chamber_volume = 6.0e-2
nozzle_throat_area = 6.0 / 1550.0
nozzle_expansion_ratio = 5.0

# Guards used in the residence-time calculation. They keep the denominator from
# going to zero before injector flow exists and prevent an unrealistically tiny
# residence time in the first few post-ignition timesteps.
minimum_combustion_mass_flow = 1.0e-6
minimum_combustion_residence_time = 1.0e-4



# -----------------------------------------------------------------------------
# Optional combustion-product map generation
# -----------------------------------------------------------------------------
# The chamber model reads gas properties from an HDF5 map so that the transient
# solve does not need to call equilibrium chemistry every timestep. The map axes
# are chamber pressure and mixture ratio. The map outputs are exactly the gas
# properties needed by the chamber Volume and the IsentropicNozzle.
if generate_combustion_map:
    map_fuel = Propellant("rp-1", temperature=fuel_temperature)
    map_ox = Propellant("lox", temperature=ox_temperature)

    def rp1_lox_products(chamber_pressure, mixture_ratio):
        # This function is called by fullplot.generate_map at every map grid
        # point. It returns equilibrium product properties for the specified
        # chamber pressure and mixture ratio.
        reactants = Reactants(
            fuels=map_fuel,
            oxidizers=map_ox,
            mixture_ratio=mixture_ratio,
        )

        gas = Equilibrium(
            reactants=reactants,
            pressure=chamber_pressure,
        )

        return {
            "temperature": gas.temperature,
            "gamma": gas.gamma,
            "gas_constant": gas.gas_constant,
            "density": gas.density,
        }

    fplt.generate_map(
        filename,
        group="products",
        axes=[
            fplt.Axis.linear(
                "chamber_pressure",
                start=ambient_pressure,
                stop=500.0 * psia_to_pa,
                count=36,
                units="Pa",
            ),
            fplt.Axis.linear(
                "mixture_ratio",
                start=mixture_ratio_map_min,
                stop=mixture_ratio_map_max,
                count=36,
            ),
        ],
        evaluate=rp1_lox_products,
        overwrite=True,
        raise_errors=True,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def command_trace(name, open_value):
    """Build a simple valve-opening command trace.

    The command starts at zero CdA, holds closed until main_valve_start_time,
    then linearly opens to the requested open CdA.
    """
    return fplt.Trace(
        x=[
            0.0,
            main_valve_start_time,
            main_valve_start_time + main_valve_open_time,
            t_final,
        ],
        y=[
            0.0,
            0.0,
            open_value,
            open_value,
        ],
        name=f"{name} Main Valve CdA Command",
        role="command",
    )


def smooth_min(a, b, eps):
    """Smooth approximation of min(a, b).

    FullFlow expressions may be used inside nonlinear residuals. A differentiable
    min avoids introducing a hard kink exactly when the wet length reaches the
    physical line length.
    """
    return 0.5 * (a + b - ((a - b) ** 2.0 + eps**2.0) ** 0.5)


def smooth_max(a, b, eps):
    """Smooth approximation of max(a, b)."""
    return 0.5 * (a + b + ((a - b) ** 2.0 + eps**2.0) ** 0.5)


def smooth_clip01(x, eps=1.0e-6):
    """Smoothly clip an expression into the interval [0, 1]."""
    return smooth_min(smooth_max(x, 0.0, eps), 1.0, eps)


def smoothstep_state(x):
    """Return a smooth 0-to-1 transition for exposure fractions.

    The polynomial x*x*(3 - 2*x) has zero slope at both endpoints. That makes
    the transition from air-exposed to liquid-exposed injector area smooth.
    """
    x = smooth_clip01(x)
    return x * x * (3.0 - 2.0 * x)


class FirstOrderProgress(Component):
    """Physical first-order combustion progress model.

    The ignition command is a latched event generated by the startup sequence.
    It represents the igniter/permissive logic saying that combustion is allowed
    to begin.

    The combustion progress is different from the ignition command. It is a
    dynamic state that estimates how much of the injected propellant stream is
    actually participating in the hot-gas chamber model:

        progress = 0  -> injected propellant is treated as unreacted drain/bypass
        progress = 1  -> injected propellant is treated as reacted hot gas

    The progress state approaches the command over the supplied residence time:

        d(progress)/dt = (command - progress) / residence_time

    In this example the residence time is calculated from chamber gas inventory
    divided by injector mass flow. Therefore the burn-in speed depends on how
    much gas is in the chamber and how much propellant is flowing through it.
    This is a reduced-order physical model, not a numerical command ramp.

    TRANSIENT_ONLY is important. The unlit steady-state solve should initialize
    the feed system. It should not drive this ignition-progress derivative to
    zero before the transient has even started.
    """

    TRANSIENT_ONLY = True

    def __init__(
        self,
        name,
        network,
        command,
        progress,
        time_constant,
    ):
        # progress_dot is a plain numeric derivative, not an assignable State.
        # FullFlow will integrate `progress` using this derivative during the
        # transient solve.
        self.progress_dot = 0.0
        self.setup()

    def evaluate_states(self):
        # Read the current command, progress, and residence-time states. Values
        # are converted to floats because this component owns a scalar ODE.
        command = float(self.command.value)
        progress = float(self.progress.value)
        time_constant = max(
            float(self.time_constant.value),
            minimum_combustion_residence_time,
        )

        # Keep the progress model in its physical range. These clamps are local
        # to the derivative calculation; the integrated State remains the solver
        # variable.
        command = min(1.0, max(0.0, command))
        progress = min(1.0, max(0.0, progress))

        # First-order residence-time response.
        self.progress_dot = float((command - progress) / time_constant)

    @property
    def dynamics(self):
        # A dynamics entry is (state, derivative). Steady-state would normally
        # drive this derivative to zero, but TRANSIENT_ONLY skips the component
        # during the pre-start steady solve.
        return [(self.progress, self.progress_dot)]


# -----------------------------------------------------------------------------
# Network and sequence
# -----------------------------------------------------------------------------
# The Network owns all components, balances, dynamic states, and tracks. The
# Sequence acts like a simple test-stand sequencer: it writes command States from
# traces and can wait on sensor conditions before a command becomes active.

Priming = Network("Dual Propellant Priming")

Startup = Sequence("Startup Sequence", Priming)

# Chamber pressure exists from the beginning because injector outlets need a
# downstream pressure boundary. Before ignition it is simply ambient. After the
# chamber Volume is added, this same State becomes the solved chamber pressure.
chamber_pressure = State(ambient_pressure)


# -----------------------------------------------------------------------------
# Fuel side
# -----------------------------------------------------------------------------
# The fuel side is a reduced-order dry-line priming model. A main valve feeds a
# growing wet liquid line. The wet liquid displaces a shared air volume, and the
# injector holes gradually transition from air-exposed to liquid-exposed area.

# Main valve command State. The Sequence writes the command trace into this
# State during the transient solve.
FuelMainValveCdaCommand = State(0.0)
fuel_main_valve_command = command_trace("Fuel", fuel_main_cda_open)

Startup.command(
    FuelMainValveCdaCommand,
    fuel_main_valve_command,
)

# Flow States. These are shared by components so the solver can connect the
# main valve, wet line, gas outlet, liquid outlet, and volumes implicitly.
fuel_main_valve_mass_flow = State(0.0)
fuel_wet_line_mass_flow = State(0.0)
fuel_injector_air_mass_flow = State(0.0)
fuel_injector_liquid_mass_flow = State(0.0)

# Pressure States. The liquid injector pressure and the trapped air pressure are
# balanced through the moving liquid/air interface balance below.
fuel_valve_outlet_pressure = State(ambient_pressure)
fuel_injector_pressure = State(ambient_pressure)
fuel_air_pressure = State(ambient_pressure)

# The liquid volume is the primary state describing the wet front. The air volume
# is whatever downstream volume is not occupied by liquid.
fuel_liquid_volume = State(initial_liquid_volume)
fuel_air_volume = total_downstream_volume - fuel_liquid_volume

# Fill fraction is based on the combined line + manifold volume. Wet line length
# is based on liquid volume / area and smoothly capped at the physical line
# length.
fuel_total_fill_fraction = fuel_liquid_volume / total_downstream_volume
fuel_raw_wet_length = fuel_liquid_volume / line_area
fuel_wet_line_length = smooth_min(fuel_raw_wet_length, line_length, wet_length_smoothing)
fuel_wetted_line_fraction = fuel_wet_line_length / line_length

# Injector exposure model. As the downstream volume fills, the injector holes are
# gradually transferred from the gas path to the liquid path.
fuel_injector_wet_ramp = (fuel_total_fill_fraction - injector_wet_start) / (injector_wet_end - injector_wet_start)
fuel_injector_liquid_exposure_fraction = smoothstep_state(fuel_injector_wet_ramp)

# Split the physical injector CdA into gas-exposed and liquid-exposed portions.
# Their sum is always fuel_injector_cda.
fuel_injector_air_exposed_cda = fuel_injector_cda * (1.0 - fuel_injector_liquid_exposure_fraction)
fuel_injector_liquid_exposed_cda = fuel_injector_cda * fuel_injector_liquid_exposure_fraction

# Thermodynamic lookup nodes for fuel liquid and trapped air. The priming liquid
# pressure is solved by the network; its temperature is fixed in this example.
FuelSourceLiquid = Lookup(
    "Fuel Source Liquid",
    Priming,
    Propellant,
    "rp-1",
    pressure=fuel_source_pressure,
    temperature=fuel_temperature,
)

FuelPrimingLiquid = Lookup(
    "Fuel Priming Liquid",
    Priming,
    Propellant,
    "rp-1",
    pressure=fuel_injector_pressure,
    temperature=fuel_temperature,
)

FuelSharedAir = Lookup(
    "Fuel Shared Air",
    Priming,
    Fluid,
    "Air",
    pressure=fuel_air_pressure,
    temperature=air_temperature,
)

# Main valve: incompressible liquid restriction from the source boundary to the
# valve outlet junction.
FuelMainValve = DischargeCoefficient(
    "Fuel Main Valve",
    Priming,
    upstream_pressure=FuelSourceLiquid.pressure,
    downstream_pressure=fuel_valve_outlet_pressure,
    density=FuelSourceLiquid.density,
    discharge_coefficient=1.0,
    cross_sectional_area=FuelMainValveCdaCommand,
    mass_flow=fuel_main_valve_mass_flow,
)

# Small junction continuity balance between the valve and wet line. No volume is
# specified, so this behaves as an algebraic flow junction.
FuelValveOutletJunction = Volume(
    "Fuel Valve Outlet Junction",
    Priming,
    pressure=fuel_valve_outlet_pressure,
    mass_flow_in=FuelMainValve.mass_flow,
    mass_flow_out=fuel_wet_line_mass_flow,
)

# Variable-length wet liquid line. The length grows as liquid fills the line.
FuelWetLiquidLine = DarcyWeisbach(
    "Fuel Wet Liquid Line",
    Priming,
    mass_flow=fuel_wet_line_mass_flow,
    upstream_pressure=fuel_valve_outlet_pressure,
    downstream_pressure=FuelPrimingLiquid.pressure,
    length=fuel_wet_line_length,
    hydraulic_diameter=line_diameter,
    cross_sectional_area=line_area,
    density=FuelPrimingLiquid.density,
    friction_factor=0.02,
)

# Gas path through the same physical injector holes before liquid arrives.
FuelInjectorAirOutlet = CompressibleOrifice(
    "Fuel Injector Air Outlet",
    Priming,
    upstream_total_pressure=FuelSharedAir.pressure,
    upstream_total_temperature=FuelSharedAir.temperature,
    downstream_pressure=ambient_pressure,
    discharge_coefficient=1.0,
    cross_sectional_area=fuel_injector_air_exposed_cda,
    gas_constant=FuelSharedAir.gas_constant,
    specific_heat_ratio=FuelSharedAir.gamma,
    upstream_static_enthalpy=FuelSharedAir.enthalpy,
    upstream_static_temperature=FuelSharedAir.temperature,
    mass_flow=fuel_injector_air_mass_flow,
)

# Liquid path through the injector holes. Downstream pressure is chamber_pressure;
# before ignition this is ambient, and after ignition it rises dynamically.
FuelInjectorLiquidOutlet = DischargeCoefficient(
    "Fuel Injector Liquid Outlet",
    Priming,
    upstream_pressure=FuelPrimingLiquid.pressure,
    downstream_pressure=chamber_pressure,
    density=FuelPrimingLiquid.density,
    discharge_coefficient=1.0,
    cross_sectional_area=fuel_injector_liquid_exposed_cda,
    mass_flow=fuel_injector_liquid_mass_flow,
)

# Liquid volume behind the injector. Its volume changes as the wet front moves.
FuelPrimingLiquidVolume = Volume(
    "Fuel Priming Liquid Volume",
    Priming,
    volume=fuel_liquid_volume,
    pressure=FuelPrimingLiquid.pressure,
    density=FuelPrimingLiquid.density,
    mass_flow_in=FuelWetLiquidLine.mass_flow,
    mass_flow_out=FuelInjectorLiquidOutlet.mass_flow,
)

# Trapped downstream air volume displaced by the incoming liquid.
FuelSharedAirVolume = Volume(
    "Fuel Shared Air Volume",
    Priming,
    volume=fuel_air_volume,
    pressure=FuelSharedAir.pressure,
    density=FuelSharedAir.density,
    mass_flow_in=0.0,
    mass_flow_out=FuelInjectorAirOutlet.mass_flow,
)

# Moving-interface balance. The liquid volume is adjusted so liquid pressure and
# trapped-air pressure match at the liquid/air interface.
FuelInterfacePressureBalance = Balance(
    "Fuel Interface Pressure Balance",
    Priming,
    variable=fuel_liquid_volume,
    function=FuelPrimingLiquid.pressure - FuelSharedAir.pressure,
)


# -----------------------------------------------------------------------------
# Ox side
# -----------------------------------------------------------------------------
# The oxidizer side mirrors the fuel-side structure, but uses LOX properties,
# source pressure, main-valve CdA, and injector CdA.

# Ox main valve command State.
OxMainValveCdaCommand = State(0.0)
ox_main_valve_command = command_trace("Ox", ox_main_cda_open)

Startup.command(
    OxMainValveCdaCommand,
    ox_main_valve_command,
)

# Ox flow States.
ox_main_valve_mass_flow = State(0.0)
ox_wet_line_mass_flow = State(0.0)
ox_injector_air_mass_flow = State(0.0)
ox_injector_liquid_mass_flow = State(0.0)

# Ox pressure States.
ox_valve_outlet_pressure = State(ambient_pressure)
ox_injector_pressure = State(ambient_pressure)
ox_air_pressure = State(ambient_pressure)

# Ox liquid/air volume split.
ox_liquid_volume = State(initial_liquid_volume)
ox_air_volume = total_downstream_volume - ox_liquid_volume

# Ox fill fraction and wet-line length.
ox_total_fill_fraction = ox_liquid_volume / total_downstream_volume
ox_raw_wet_length = ox_liquid_volume / line_area
ox_wet_line_length = smooth_min(ox_raw_wet_length, line_length, wet_length_smoothing)
ox_wetted_line_fraction = ox_wet_line_length / line_length

# Ox injector exposure model.
ox_injector_wet_ramp = (ox_total_fill_fraction - injector_wet_start) / (injector_wet_end - injector_wet_start)
ox_injector_liquid_exposure_fraction = smoothstep_state(ox_injector_wet_ramp)

# Split the physical ox injector CdA into gas-exposed and liquid-exposed portions.
ox_injector_air_exposed_cda = ox_injector_cda * (1.0 - ox_injector_liquid_exposure_fraction)
ox_injector_liquid_exposed_cda = ox_injector_cda * ox_injector_liquid_exposure_fraction

# Thermodynamic lookup nodes for LOX and trapped air.
OxSourceLiquid = Lookup(
    "Ox Source Liquid",
    Priming,
    Propellant,
    "lox",
    pressure=ox_source_pressure,
    temperature=ox_temperature,
)

OxPrimingLiquid = Lookup(
    "Ox Priming Liquid",
    Priming,
    Propellant,
    "lox",
    pressure=ox_injector_pressure,
    temperature=ox_temperature,
)

OxSharedAir = Lookup(
    "Ox Shared Air",
    Priming,
    Fluid,
    "Air",
    pressure=ox_air_pressure,
    temperature=air_temperature,
)

# Ox main valve restriction.
OxMainValve = DischargeCoefficient(
    "Ox Main Valve",
    Priming,
    upstream_pressure=OxSourceLiquid.pressure,
    downstream_pressure=ox_valve_outlet_pressure,
    density=OxSourceLiquid.density,
    discharge_coefficient=1.0,
    cross_sectional_area=OxMainValveCdaCommand,
    mass_flow=ox_main_valve_mass_flow,
)

# Ox valve outlet algebraic junction.
OxValveOutletJunction = Volume(
    "Ox Valve Outlet Junction",
    Priming,
    pressure=ox_valve_outlet_pressure,
    mass_flow_in=OxMainValve.mass_flow,
    mass_flow_out=ox_wet_line_mass_flow,
)

# Ox variable-length wet liquid line.
OxWetLiquidLine = DarcyWeisbach(
    "Ox Wet Liquid Line",
    Priming,
    mass_flow=ox_wet_line_mass_flow,
    upstream_pressure=ox_valve_outlet_pressure,
    downstream_pressure=OxPrimingLiquid.pressure,
    length=ox_wet_line_length,
    hydraulic_diameter=line_diameter,
    cross_sectional_area=line_area,
    density=OxPrimingLiquid.density,
    friction_factor=0.02,
)

# Ox-side trapped-air path through the open injector holes.
OxInjectorAirOutlet = CompressibleOrifice(
    "Ox Injector Air Outlet",
    Priming,
    upstream_total_pressure=OxSharedAir.pressure,
    upstream_total_temperature=OxSharedAir.temperature,
    downstream_pressure=ambient_pressure,
    discharge_coefficient=1.0,
    cross_sectional_area=ox_injector_air_exposed_cda,
    gas_constant=OxSharedAir.gas_constant,
    specific_heat_ratio=OxSharedAir.gamma,
    upstream_static_enthalpy=OxSharedAir.enthalpy,
    upstream_static_temperature=OxSharedAir.temperature,
    mass_flow=ox_injector_air_mass_flow,
)

# LOX injector liquid path into the chamber pressure boundary.
OxInjectorLiquidOutlet = DischargeCoefficient(
    "Ox Injector Liquid Outlet",
    Priming,
    upstream_pressure=OxPrimingLiquid.pressure,
    downstream_pressure=chamber_pressure,
    density=OxPrimingLiquid.density,
    discharge_coefficient=1.0,
    cross_sectional_area=ox_injector_liquid_exposed_cda,
    mass_flow=ox_injector_liquid_mass_flow,
)

# Ox liquid fill volume.
OxPrimingLiquidVolume = Volume(
    "Ox Priming Liquid Volume",
    Priming,
    volume=ox_liquid_volume,
    pressure=OxPrimingLiquid.pressure,
    density=OxPrimingLiquid.density,
    mass_flow_in=OxWetLiquidLine.mass_flow,
    mass_flow_out=OxInjectorLiquidOutlet.mass_flow,
)

# Ox trapped-air volume.
OxSharedAirVolume = Volume(
    "Ox Shared Air Volume",
    Priming,
    volume=ox_air_volume,
    pressure=OxSharedAir.pressure,
    density=OxSharedAir.density,
    mass_flow_in=0.0,
    mass_flow_out=OxInjectorAirOutlet.mass_flow,
)

# Ox liquid/air interface pressure balance.
OxInterfacePressureBalance = Balance(
    "Ox Interface Pressure Balance",
    Priming,
    variable=ox_liquid_volume,
    function=OxPrimingLiquid.pressure - OxSharedAir.pressure,
)


# -----------------------------------------------------------------------------
# Ignition permissive sensors and bluelines
# -----------------------------------------------------------------------------
# The bluelines represent minimum injector pressures required before ignition is
# allowed. The sensors latch these events during the transient.
FIPT_blueline = fplt.Trace(
    x=[0.0, t_final],
    y=[180.0, 180.0],
    name="FIPT Ignition Blueline",
    role="blueline",
)

OIPT_blueline = fplt.Trace(
    x=[0.0, t_final],
    y=[90.0, 90.0],
    name="OIPT Ignition Blueline",
    role="blueline",
)


# FIPT/OIPT are simulated test-stand pressure measurements. They read the model
# injector pressures and compare them against the blueline traces.
FIPT = Sensor(
    "FIPT",
    Priming,
    reading=FuelPrimingLiquid.pressure / psia_to_pa,
    conditions=FIPT_blueline,
)

OIPT = Sensor(
    "OIPT",
    Priming,
    reading=OxPrimingLiquid.pressure / psia_to_pa,
    conditions=OIPT_blueline,
)


# -----------------------------------------------------------------------------
# Ignition command
# -----------------------------------------------------------------------------
# The ignition command is a sequence output, not a chamber-pressure command. It
# turns on only after both pressure bluelines latch. The physical chamber response
# still comes from the combustion-progress state, chamber mass balance, and nozzle
# flow.

# Latched ignition permissive. It is 0 before both sensors satisfy their
# bluelines and 1 after they have latched.
ignition_command = State(0.0)

ignition_command_trace = fplt.Trace.constant(
    "Ignition Command",
    x=fuel_main_valve_command.x,
    y=1.0,
    role="command",
)

Startup.command(
    ignition_command,
    ignition_command_trace,
    condition=[
        (FIPT, "FIPT Ignition Blueline"),
        (OIPT, "OIPT Ignition Blueline"),
    ],
)


# -----------------------------------------------------------------------------
# Steady-state initialization
# -----------------------------------------------------------------------------

# Initialize the unlit feed-system state first. The chamber/nozzle/combustion
# dynamics are added after this steady solve so pre-start trim does not force a
# fictitious combustion steady-state point.
#
# This ordering is intentional for startup examples:
#
#     - the steady solve finds the closed-valve, unlit feed-system state;
#     - the transient solve then adds ignition/chamber physics and runs forward;
#     - the combustion-progress component is TRANSIENT_ONLY, so it represents a
#       true startup dynamic rather than a steady trim target.
SteadyState(Priming).solve(
    verbose=True,
    ignore_balances=[
        "Fuel Interface Pressure Balance",
        "Ox Interface Pressure Balance",
    ],
    filename=filename,
)


# -----------------------------------------------------------------------------
# Chamber, nozzle, and physical combustion progress
# -----------------------------------------------------------------------------
# These components are appended after the unlit steady-state initialization.
# Chamber pressure is the same State already used as the injector downstream
# pressure, so injector flow naturally responds when the chamber pressure rises.

# Mixture ratio is held at the lower map bound until combustion progress exists.
# Once burning begins, a Sequence updates it from the liquid injector flow ratio.
mixture_ratio = State(mixture_ratio_map_min)

# Combustion-product properties. The map outputs are used by both the chamber
# inventory calculation and the nozzle.
ChamberGasMap = Map.from_hdf5(
    "Chamber Gas Map",
    Priming,
    filename,
    group="products",
    inputs={
        "chamber_pressure": chamber_pressure,
        "mixture_ratio": mixture_ratio,
    },
)

# Total liquid propellant entering the injector/chamber control volume.
injector_total_mass_flow = (
    OxInjectorLiquidOutlet.mass_flow
    + FuelInjectorLiquidOutlet.mass_flow
)

# Approximate hot-gas mass inventory in the chamber if it were filled with the
# current combustion products.
chamber_gas_inventory = ChamberGasMap.density * chamber_volume

# Residence time estimate. More chamber inventory or less injector flow gives a
# slower combustion-progress response; more injector flow gives a faster response.
combustion_residence_time = chamber_gas_inventory / smooth_max(
    injector_total_mass_flow,
    minimum_combustion_mass_flow,
    1.0e-8,
)

# Apply a lower bound to keep the reduced-order progress model well behaved at
# the first few timesteps after ignition.
combustion_residence_time = smooth_max(
    combustion_residence_time,
    minimum_combustion_residence_time,
    1.0e-9,
)

# Dynamic burn-progress state. Zero means no injected propellant is contributing
# to the hot-gas chamber model. One means all injected propellant is treated as
# reacted chamber gas.
combustion_progress = State(0.0)

CombustionProgress = FirstOrderProgress(
    "Combustion Progress",
    Priming,
    command=ignition_command,
    progress=combustion_progress,
    time_constant=combustion_residence_time,
)


def mixture_ratio_update(t, progress, ox_mdot, fuel_mdot):
    """Update the map mixture ratio from injector liquid flows.

    Before combustion progress is established, the map is held at its lower MR
    bound so the chamber gas lookup remains inside its valid domain. Once the
    model is burning, MR follows the oxidizer/fuel injector mass-flow ratio and
    is clipped to the map range.
    """
    if progress < 1.0e-3:
        return mixture_ratio_map_min

    if fuel_mdot <= 0.0:
        return mixture_ratio_map_min

    value = ox_mdot / fuel_mdot

    if value < mixture_ratio_map_min:
        return mixture_ratio_map_min

    if value > mixture_ratio_map_max:
        return mixture_ratio_map_max

    return value


MixtureRatioUpdate = Sequence(
    "Mixture Ratio Update",
    Priming,
    target=mixture_ratio,
    function=mixture_ratio_update,
    inputs=[
        combustion_progress,
        OxInjectorLiquidOutlet.mass_flow,
        FuelInjectorLiquidOutlet.mass_flow,
    ],
)

# Reduced-order burned/unburned split. This is the central startup model:
# combustion_progress decides what fraction of injected propellant contributes
# to hot-gas chamber pressurization.
reacted_propellant_mass_flow = combustion_progress * injector_total_mass_flow
unreacted_propellant_mass_flow = (1.0 - combustion_progress) * injector_total_mass_flow

# Nozzle flow is calculated from chamber stagnation pressure, chamber gas
# properties, throat area, expansion ratio, and ambient pressure. It is not
# prescribed by a command trace.
Nozzle = IsentropicNozzle(
    "Nozzle",
    Priming,
    upstream_total_pressure=chamber_pressure,
    upstream_total_temperature=ChamberGasMap.temperature,
    ambient_pressure=ambient_pressure,
    specific_heat_ratio=ChamberGasMap.gamma,
    gas_constant=ChamberGasMap.gas_constant,
    throat_area=nozzle_throat_area,
    expansion_ratio=nozzle_expansion_ratio,
)

# Effective external mass flow for plotting. Before ignition this is mostly
# unreacted propellant drain/bypass. After ignition it becomes nozzle flow.
effective_exit_mass_flow = unreacted_propellant_mass_flow + Nozzle.mass_flow

# Chamber hot-gas mass balance. Reacted propellant adds mass to the chamber;
# the nozzle removes mass. This mass balance is what raises chamber pressure.
Chamber = Volume(
    "Combustion Chamber",
    Priming,
    pressure=chamber_pressure,
    volume=chamber_volume,
    density=ChamberGasMap.density,
    mass_flow_in=reacted_propellant_mass_flow,
    mass_flow_out=Nozzle.mass_flow,
)


# -----------------------------------------------------------------------------
# Tracks
# -----------------------------------------------------------------------------
# Tracks are stored in the HDF5 result file and plotted below. The names include
# units so they can be used directly as plot labels or inspected in FullPlot.

Priming.track("Fuel Main Valve CdA Command [m2]", FuelMainValveCdaCommand)
Priming.track("Ox Main Valve CdA Command [m2]", OxMainValveCdaCommand)

Priming.track("Fuel Valve Outlet Pressure [psia]", fuel_valve_outlet_pressure / psia_to_pa)
Priming.track("Fuel Injector Pressure [psia]", FuelPrimingLiquid.pressure / psia_to_pa)
Priming.track("Ox Valve Outlet Pressure [psia]", ox_valve_outlet_pressure / psia_to_pa)
Priming.track("Ox Injector Pressure [psia]", OxPrimingLiquid.pressure / psia_to_pa)
Priming.track("Chamber Pressure [psia]", chamber_pressure / psia_to_pa)

Priming.track("Fuel Inlet Mass Flow [kg/s]", FuelWetLiquidLine.mass_flow)
Priming.track("Fuel Injector Air Outlet Mass Flow [kg/s]", FuelInjectorAirOutlet.mass_flow)
Priming.track("Fuel Injector Liquid Outlet Mass Flow [kg/s]", FuelInjectorLiquidOutlet.mass_flow)

Priming.track("Ox Inlet Mass Flow [kg/s]", OxWetLiquidLine.mass_flow)
Priming.track("Ox Injector Air Outlet Mass Flow [kg/s]", OxInjectorAirOutlet.mass_flow)
Priming.track("Ox Injector Liquid Outlet Mass Flow [kg/s]", OxInjectorLiquidOutlet.mass_flow)

Priming.track("Injector Total Mass Flow [kg/s]", injector_total_mass_flow)
Priming.track("Reacted Propellant Mass Flow [kg/s]", reacted_propellant_mass_flow)
Priming.track("Unreacted Propellant Mass Flow [kg/s]", unreacted_propellant_mass_flow)
Priming.track("Effective Exit Mass Flow [kg/s]", effective_exit_mass_flow)
Priming.track("Nozzle Mass Flow [kg/s]", Nozzle.mass_flow)

Priming.track("Fuel Filled Volume [in3]", fuel_liquid_volume / in3_to_m3)
Priming.track("Ox Filled Volume [in3]", ox_liquid_volume / in3_to_m3)

Priming.track("Fuel Total Fill Fraction [-]", fuel_total_fill_fraction)
Priming.track("Fuel Wetted Line Fraction [-]", fuel_wetted_line_fraction)
Priming.track("Fuel Injector Liquid Exposure Fraction [-]", fuel_injector_liquid_exposure_fraction)

Priming.track("Ox Total Fill Fraction [-]", ox_total_fill_fraction)
Priming.track("Ox Wetted Line Fraction [-]", ox_wetted_line_fraction)
Priming.track("Ox Injector Liquid Exposure Fraction [-]", ox_injector_liquid_exposure_fraction)

Priming.track("Fuel Injector Air-Exposed CdA [m2]", fuel_injector_air_exposed_cda)
Priming.track("Fuel Injector Liquid-Exposed CdA [m2]", fuel_injector_liquid_exposed_cda)
Priming.track("Ox Injector Air-Exposed CdA [m2]", ox_injector_air_exposed_cda)
Priming.track("Ox Injector Liquid-Exposed CdA [m2]", ox_injector_liquid_exposed_cda)

Priming.track("Ignition Command [-]", ignition_command)
Priming.track("Combustion Progress [-]", combustion_progress)
Priming.track("Combustion Residence Time [s]", combustion_residence_time)
Priming.track("Mixture Ratio", mixture_ratio)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

Transient(Priming).solve(
    dt=dt,
    t_final=t_final,
    verbose=True,
    statistics=True,
    filename=filename,
)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
# The plots are grouped by physical meaning rather than by component. This makes
# the example read like a startup data review: commands, pressures, flows,
# fill/exposure state, mixture ratio, ignition progress, and residence time.

result = fplt.open(filename).at(
    "Dual_Propellant_Priming/transient/runs/base/tracks"
)

# Main valve command review.
fuel_valve_cda = result.trace(
    y="Fuel Main Valve CdA Command [m2]",
    x="time",
    name="Fuel Main Valve CdA",
    role="command",
)

ox_valve_cda = result.trace(
    y="Ox Main Valve CdA Command [m2]",
    x="time",
    name="Ox Main Valve CdA",
    role="command",
)

result.plot(
    y=[fuel_valve_cda, ox_valve_cda],
    xlabel="Time [s]",
    ylabel="Main valve CdA [m2]",
    title="Main Valve Commands",
)

# Feed-system and chamber pressure review.
fuel_valve_outlet_pressure = result.trace(
    y="Fuel Valve Outlet Pressure [psia]",
    x="time",
    name="Fuel Valve Outlet",
)

fuel_injector_pressure = result.trace(
    y="Fuel Injector Pressure [psia]",
    x="time",
    name="Fuel Injector",
)

ox_valve_outlet_pressure = result.trace(
    y="Ox Valve Outlet Pressure [psia]",
    x="time",
    name="Ox Valve Outlet",
)

ox_injector_pressure = result.trace(
    y="Ox Injector Pressure [psia]",
    x="time",
    name="Ox Injector",
)

chamber_pressure_trace = result.trace(
    y="Chamber Pressure [psia]",
    x="time",
    name="Chamber Pressure",
)

result.plot(
    y=[
        fuel_valve_outlet_pressure,
        fuel_injector_pressure,
        ox_valve_outlet_pressure,
        ox_injector_pressure,
        chamber_pressure_trace,
        FIPT_blueline,
        OIPT_blueline,
    ],
    xlabel="Time [s]",
    ylabel="Pressure [psia]",
    title="Valve Outlet, Injector, and Chamber Pressures",
)

# Liquid mass-flow review through the wet lines and injector liquid paths.
fuel_inlet_mdot = result.trace(
    y="Fuel Inlet Mass Flow [kg/s]",
    x="time",
    name="Fuel Inlet",
)

fuel_liquid_out_mdot = result.trace(
    y="Fuel Injector Liquid Outlet Mass Flow [kg/s]",
    x="time",
    name="Fuel Injector Liquid Outlet",
)

ox_inlet_mdot = result.trace(
    y="Ox Inlet Mass Flow [kg/s]",
    x="time",
    name="Ox Inlet",
)

ox_liquid_out_mdot = result.trace(
    y="Ox Injector Liquid Outlet Mass Flow [kg/s]",
    x="time",
    name="Ox Injector Liquid Outlet",
)

result.plot(
    y=[
        fuel_inlet_mdot,
        fuel_liquid_out_mdot,
        ox_inlet_mdot,
        ox_liquid_out_mdot,
    ],
    xlabel="Time [s]",
    ylabel="Liquid mass flow [kg/s]",
    title="Liquid Inlet and Injector Outlet Mass Flows",
)

# Air blowdown/venting during line priming.
fuel_air_out_mdot = result.trace(
    y="Fuel Injector Air Outlet Mass Flow [kg/s]",
    x="time",
    name="Fuel Injector Air Outlet",
)

ox_air_out_mdot = result.trace(
    y="Ox Injector Air Outlet Mass Flow [kg/s]",
    x="time",
    name="Ox Injector Air Outlet",
)

result.plot(
    y=[fuel_air_out_mdot, ox_air_out_mdot],
    xlabel="Time [s]",
    ylabel="Air mass flow [kg/s]",
    title="Injector Air Outlet Mass Flow",
)

# Burned/unburned split and nozzle buildup.
injector_total_mdot = result.trace(
    y="Injector Total Mass Flow [kg/s]",
    x="time",
    name="Injector Total",
)

reacted_mdot = result.trace(
    y="Reacted Propellant Mass Flow [kg/s]",
    x="time",
    name="Reacted Propellant",
)

unreacted_mdot = result.trace(
    y="Unreacted Propellant Mass Flow [kg/s]",
    x="time",
    name="Unreacted Propellant",
)

exit_mdot = result.trace(
    y="Effective Exit Mass Flow [kg/s]",
    x="time",
    name="Effective Exit",
)

nozzle_mdot = result.trace(
    y="Nozzle Mass Flow [kg/s]",
    x="time",
    name="Nozzle",
)

result.plot(
    y=[injector_total_mdot, reacted_mdot, unreacted_mdot, exit_mdot, nozzle_mdot],
    xlabel="Time [s]",
    ylabel="Mass flow [kg/s]",
    title="Injector, Combustion, Bypass, and Nozzle Flows",
)

# Filled volume and line wetting diagnostics.
fuel_filled_volume = result.trace(
    y="Fuel Filled Volume [in3]",
    x="time",
    name="Fuel Filled Volume",
)

ox_filled_volume = result.trace(
    y="Ox Filled Volume [in3]",
    x="time",
    name="Ox Filled Volume",
)

result.plot(
    y=[fuel_filled_volume, ox_filled_volume],
    xlabel="Time [s]",
    ylabel="Filled volume [in3]",
    title="Filled Downstream Volume",
)

fuel_total_fill = result.trace(
    y="Fuel Total Fill Fraction [-]",
    x="time",
    name="Fuel Total Fill",
)

ox_total_fill = result.trace(
    y="Ox Total Fill Fraction [-]",
    x="time",
    name="Ox Total Fill",
)

fuel_wetted_line = result.trace(
    y="Fuel Wetted Line Fraction [-]",
    x="time",
    name="Fuel Wetted Line",
)

ox_wetted_line = result.trace(
    y="Ox Wetted Line Fraction [-]",
    x="time",
    name="Ox Wetted Line",
)

fuel_injector_exposure = result.trace(
    y="Fuel Injector Liquid Exposure Fraction [-]",
    x="time",
    name="Fuel Injector Liquid Exposure",
)

ox_injector_exposure = result.trace(
    y="Ox Injector Liquid Exposure Fraction [-]",
    x="time",
    name="Ox Injector Liquid Exposure",
)

result.plot(
    y=[
        fuel_total_fill,
        ox_total_fill,
        fuel_wetted_line,
        ox_wetted_line,
        fuel_injector_exposure,
        ox_injector_exposure,
    ],
    xlabel="Time [s]",
    ylabel="Fraction [-]",
    title="Fill, Line Wetting, and Injector Liquid Exposure",
)

# Injector CdA split between gas and liquid exposure.
fuel_air_cda = result.trace(
    y="Fuel Injector Air-Exposed CdA [m2]",
    x="time",
    name="Fuel Air-Exposed CdA",
)

fuel_liquid_cda = result.trace(
    y="Fuel Injector Liquid-Exposed CdA [m2]",
    x="time",
    name="Fuel Liquid-Exposed CdA",
)

ox_air_cda = result.trace(
    y="Ox Injector Air-Exposed CdA [m2]",
    x="time",
    name="Ox Air-Exposed CdA",
)

ox_liquid_cda = result.trace(
    y="Ox Injector Liquid-Exposed CdA [m2]",
    x="time",
    name="Ox Liquid-Exposed CdA",
)

result.plot(
    y=[fuel_air_cda, fuel_liquid_cda, ox_air_cda, ox_liquid_cda],
    xlabel="Time [s]",
    ylabel="Injector phase-exposed CdA [m2]",
    title="Injector Phase-Exposed CdA",
)

# Chamber map mixture-ratio input.
mr = result.trace(
    y="Mixture Ratio",
    x="time",
    name="Mixture Ratio",
)

result.plot(
    y=mr,
    xlabel="Time [s]",
    ylabel="Mixture Ratio",
    title="Startup MR",
)

# Ignition command versus physical combustion progress. The command is latched;
# progress is dynamic.
ignition_command_plot = result.trace(
    y="Ignition Command [-]",
    x="time",
    name="Ignition Command",
)

combustion_progress_plot = result.trace(
    y="Combustion Progress [-]",
    x="time",
    name="Combustion Progress",
)

result.plot(
    y=[ignition_command_plot, combustion_progress_plot],
    xlabel="Time [s]",
    ylabel="Fraction [-]",
    title="Ignition Command and Combustion Progress",
)

# Residence time driving the progress model.
residence_time_plot = result.trace(
    y="Combustion Residence Time [s]",
    x="time",
    name="Residence Time",
)

result.plot(
    y=residence_time_plot,
    xlabel="Time [s]",
    ylabel="Time [s]",
    title="Combustion Residence Time",
)

fplt.show()
