import numpy as np

from fullflow import *
from thermoprop import *
import fullplot as fplt


"""
Dual-Propellant Dry-Line Priming Test
=====================================

This test models startup priming of two initially air-filled propellant paths:

    - RP-1 fuel side
    - LOX oxidizer side

There is no chamber model yet. Both injector outlets discharge to ambient.

Each side is modeled as:

    primed source -> main valve command -> valve outlet node
                  -> variable-length wet liquid line
                  -> injector/manifold pressure node
                  -> shared downstream air volume
                  -> open injector orifices -> ambient

The injector orifices are always open. They are not scheduled valves.

Before liquid breakthrough, the open injector holes are exposed mostly to air.
After breakthrough, the same open holes are exposed mostly to liquid. The model
represents that with gas-exposed and liquid-exposed injector CdA values whose
sum is always the physical injector CdA.

The only command traces in this model are the upstream main valve CdA commands.
"""
filename = 'test'
generate_combustion_map = False

mixture_ratio_map_min = 0.5
mixture_ratio_map_max = 6.0

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

psia_to_pa = 6894.76
in_to_m = 1.0 / 39.37
in3_to_m3 = in_to_m**3

dt = 0.001
t_final = 0.35

ambient_pressure = 14.67 * psia_to_pa

fuel_source_pressure = 450.0 * psia_to_pa
ox_source_pressure = 400.0 * psia_to_pa

fuel_temperature = 300.0
ox_temperature = 90.0
air_temperature = 300.0

line_length = 0.30
line_area = 0.5e-4
line_diameter = np.sqrt(4.0 * line_area / np.pi)
line_volume = line_area * line_length

manifold_volume = 20.0 * in3_to_m3
total_downstream_volume = line_volume + manifold_volume

fuel_main_cda_open = 0.5e-4
ox_main_cda_open = 1.0e-4

fuel_injector_cda = 0.5e-4
ox_injector_cda = 1.0e-4

initial_wet_length = 0.02
initial_liquid_volume = initial_wet_length * line_area

wet_length_smoothing = 1.0e-5

main_valve_start_time = 0.05
main_valve_open_time = 0.05

injector_wet_start = 0.90
injector_wet_end = 0.99





if generate_combustion_map:
    map_fuel = Propellant("rp-1", temperature=fuel_temperature)
    map_ox = Propellant("lox", temperature=ox_temperature)

    def rp1_lox_products(chamber_pressure, mixture_ratio):
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
    return 0.5 * (a + b - ((a - b) ** 2.0 + eps**2.0) ** 0.5)


def smooth_max(a, b, eps):
    return 0.5 * (a + b + ((a - b) ** 2.0 + eps**2.0) ** 0.5)


def smooth_clip01(x, eps=1.0e-6):
    return smooth_min(smooth_max(x, 0.0, eps), 1.0, eps)


def smoothstep_state(x):
    x = smooth_clip01(x)
    return x * x * (3.0 - 2.0 * x)


# -----------------------------------------------------------------------------
# Network and sequence
# -----------------------------------------------------------------------------

Priming = Network("Dual Propellant Priming")



Startup = Sequence("Startup Sequence", Priming)


# -----------------------------------------------------------------------------
# Fuel side
# -----------------------------------------------------------------------------

FuelMainValveCdaCommand = State(0.0)
fuel_main_valve_command = command_trace("Fuel", fuel_main_cda_open)

Startup.command(
    FuelMainValveCdaCommand,
    fuel_main_valve_command,
)

fuel_main_valve_mass_flow = State(0.0)
fuel_wet_line_mass_flow = State(0.0)
fuel_injector_air_mass_flow = State(0.0)
fuel_injector_liquid_mass_flow = State(0.0)

fuel_valve_outlet_pressure = State(ambient_pressure)
fuel_injector_pressure = State(ambient_pressure)
fuel_air_pressure = State(ambient_pressure)

fuel_liquid_volume = State(initial_liquid_volume)
fuel_air_volume = total_downstream_volume - fuel_liquid_volume

fuel_total_fill_fraction = fuel_liquid_volume / total_downstream_volume
fuel_raw_wet_length = fuel_liquid_volume / line_area
fuel_wet_line_length = smooth_min(fuel_raw_wet_length, line_length, wet_length_smoothing)
fuel_wetted_line_fraction = fuel_wet_line_length / line_length

fuel_injector_wet_ramp = (fuel_total_fill_fraction - injector_wet_start) / (injector_wet_end - injector_wet_start)
fuel_injector_liquid_exposure_fraction = smoothstep_state(fuel_injector_wet_ramp)

fuel_injector_air_exposed_cda = fuel_injector_cda * (1.0 - fuel_injector_liquid_exposure_fraction)
fuel_injector_liquid_exposed_cda = fuel_injector_cda * fuel_injector_liquid_exposure_fraction

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

FuelValveOutletJunction = Volume(
    "Fuel Valve Outlet Junction",
    Priming,
    pressure=fuel_valve_outlet_pressure,
    mass_flow_in=FuelMainValve.mass_flow,
    mass_flow_out=fuel_wet_line_mass_flow,
)

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

FuelInjectorLiquidOutlet = DischargeCoefficient(
    "Fuel Injector Liquid Outlet",
    Priming,
    upstream_pressure=FuelPrimingLiquid.pressure,
    downstream_pressure=ambient_pressure,
    density=FuelPrimingLiquid.density,
    discharge_coefficient=1.0,
    cross_sectional_area=fuel_injector_liquid_exposed_cda,
    mass_flow=fuel_injector_liquid_mass_flow,
)

FuelPrimingLiquidVolume = Volume(
    "Fuel Priming Liquid Volume",
    Priming,
    volume=fuel_liquid_volume,
    pressure=FuelPrimingLiquid.pressure,
    density=FuelPrimingLiquid.density,
    mass_flow_in=FuelWetLiquidLine.mass_flow,
    mass_flow_out=FuelInjectorLiquidOutlet.mass_flow,
)

FuelSharedAirVolume = Volume(
    "Fuel Shared Air Volume",
    Priming,
    volume=fuel_air_volume,
    pressure=FuelSharedAir.pressure,
    density=FuelSharedAir.density,
    mass_flow_in=0.0,
    mass_flow_out=FuelInjectorAirOutlet.mass_flow,
)

FuelInterfacePressureBalance = Balance(
    "Fuel Interface Pressure Balance",
    Priming,
    variable=fuel_liquid_volume,
    function=FuelPrimingLiquid.pressure - FuelSharedAir.pressure,
)


# -----------------------------------------------------------------------------
# Ox side
# -----------------------------------------------------------------------------

OxMainValveCdaCommand = State(0.0)
ox_main_valve_command = command_trace("Ox", ox_main_cda_open)

Startup.command(
    OxMainValveCdaCommand,
    ox_main_valve_command,
)

ox_main_valve_mass_flow = State(0.0)
ox_wet_line_mass_flow = State(0.0)
ox_injector_air_mass_flow = State(0.0)
ox_injector_liquid_mass_flow = State(0.0)

ox_valve_outlet_pressure = State(ambient_pressure)
ox_injector_pressure = State(ambient_pressure)
ox_air_pressure = State(ambient_pressure)

ox_liquid_volume = State(initial_liquid_volume)
ox_air_volume = total_downstream_volume - ox_liquid_volume

ox_total_fill_fraction = ox_liquid_volume / total_downstream_volume
ox_raw_wet_length = ox_liquid_volume / line_area
ox_wet_line_length = smooth_min(ox_raw_wet_length, line_length, wet_length_smoothing)
ox_wetted_line_fraction = ox_wet_line_length / line_length

ox_injector_wet_ramp = (ox_total_fill_fraction - injector_wet_start) / (injector_wet_end - injector_wet_start)
ox_injector_liquid_exposure_fraction = smoothstep_state(ox_injector_wet_ramp)

ox_injector_air_exposed_cda = ox_injector_cda * (1.0 - ox_injector_liquid_exposure_fraction)
ox_injector_liquid_exposed_cda = ox_injector_cda * ox_injector_liquid_exposure_fraction

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

OxValveOutletJunction = Volume(
    "Ox Valve Outlet Junction",
    Priming,
    pressure=ox_valve_outlet_pressure,
    mass_flow_in=OxMainValve.mass_flow,
    mass_flow_out=ox_wet_line_mass_flow,
)

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

OxInjectorLiquidOutlet = DischargeCoefficient(
    "Ox Injector Liquid Outlet",
    Priming,
    upstream_pressure=OxPrimingLiquid.pressure,
    downstream_pressure=ambient_pressure,
    density=OxPrimingLiquid.density,
    discharge_coefficient=1.0,
    cross_sectional_area=ox_injector_liquid_exposed_cda,
    mass_flow=ox_injector_liquid_mass_flow,
)

OxPrimingLiquidVolume = Volume(
    "Ox Priming Liquid Volume",
    Priming,
    volume=ox_liquid_volume,
    pressure=OxPrimingLiquid.pressure,
    density=OxPrimingLiquid.density,
    mass_flow_in=OxWetLiquidLine.mass_flow,
    mass_flow_out=OxInjectorLiquidOutlet.mass_flow,
)

OxSharedAirVolume = Volume(
    "Ox Shared Air Volume",
    Priming,
    volume=ox_air_volume,
    pressure=OxSharedAir.pressure,
    density=OxSharedAir.density,
    mass_flow_in=0.0,
    mass_flow_out=OxInjectorAirOutlet.mass_flow,
)

OxInterfacePressureBalance = Balance(
    "Ox Interface Pressure Balance",
    Priming,
    variable=ox_liquid_volume,
    function=OxPrimingLiquid.pressure - OxSharedAir.pressure,
)


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


mixture_ratio = OxInjectorLiquidOutlet.mass_flow / FuelInjectorLiquidOutlet.mass_flow


# -----------------------------------------------------------------------------
# Tracks
# -----------------------------------------------------------------------------

Priming.track("Fuel Main Valve CdA Command [m2]", FuelMainValveCdaCommand)
Priming.track("Ox Main Valve CdA Command [m2]", OxMainValveCdaCommand)

Priming.track("Fuel Valve Outlet Pressure [psia]", fuel_valve_outlet_pressure / psia_to_pa)
Priming.track("Fuel Injector Pressure [psia]", FuelPrimingLiquid.pressure / psia_to_pa)
Priming.track("Ox Valve Outlet Pressure [psia]", ox_valve_outlet_pressure / psia_to_pa)
Priming.track("Ox Injector Pressure [psia]", OxPrimingLiquid.pressure / psia_to_pa)

Priming.track("Fuel Inlet Mass Flow [kg/s]", FuelWetLiquidLine.mass_flow)
Priming.track("Fuel Injector Air Outlet Mass Flow [kg/s]", FuelInjectorAirOutlet.mass_flow)
Priming.track("Fuel Injector Liquid Outlet Mass Flow [kg/s]", FuelInjectorLiquidOutlet.mass_flow)

Priming.track("Ox Inlet Mass Flow [kg/s]", OxWetLiquidLine.mass_flow)
Priming.track("Ox Injector Air Outlet Mass Flow [kg/s]", OxInjectorAirOutlet.mass_flow)
Priming.track("Ox Injector Liquid Outlet Mass Flow [kg/s]", OxInjectorLiquidOutlet.mass_flow)

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

Priming.track("Mixture Ratio", mixture_ratio)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------


SteadyState(Priming).solve(
    verbose=True,
    ignore_balances=[
        "Fuel Interface Pressure Balance",
        "Ox Interface Pressure Balance",
    ],
    filename=filename,
)

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

result = fplt.open(filename).at(
    "Dual_Propellant_Priming/transient/runs/base/tracks"
)

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

result.plot(
    y=[
        fuel_valve_outlet_pressure,
        fuel_injector_pressure,
        ox_valve_outlet_pressure,
        ox_injector_pressure,
    ],
    xlabel="Time [s]",
    ylabel="Pressure [psia]",
    title="Valve Outlet and Injector Pressures",
)

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



fplt.show()