from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np

from fullflow import *
from thermoprop import *


psia_to_pa = 6894.76
filename = "heatsink_tca"

make_map = False


use_multiprocessing = True
map_workers = min(4, os.cpu_count() or 1)
map_chunksize = 25


# ---------------------------------------------------------------------------
# Multiprocessing map helper
# ---------------------------------------------------------------------------

def _evaluate_map_job(job):
    evaluate, axis_names, axis_values, constants, input_names, index = job

    inputs = dict(constants)

    for name, values, i in zip(axis_names, axis_values, index):
        inputs[name] = float(values[i])

    key = tuple(inputs[name] for name in input_names)

    return key, evaluate(**inputs)


def generate_map_mp(
    filename,
    group,
    axes,
    evaluate,
    constants=None,
    workers=4,
    chunksize=25,
    **kwargs,
):
    constants = {} if constants is None else dict(constants)

    axis_names = [axis.name for axis in axes]
    axis_values = [axis.values for axis in axes]
    input_names = tuple(list(constants.keys()) + axis_names)

    shape = tuple(len(values) for values in axis_values)

    jobs = [
        (evaluate, axis_names, axis_values, constants, input_names, index)
        for index in np.ndindex(shape)
    ]

    print(f"{group}: evaluating {len(jobs)} map points with {workers} workers...")

    precomputed = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for counter, (key, values) in enumerate(pool.map(_evaluate_map_job, jobs, chunksize=chunksize), start=1):
            precomputed[key] = values

            if counter % 100 == 0 or counter == len(jobs):
                print(f"{group}: completed {counter} / {len(jobs)}")

    def precomputed_map(**inputs):
        key = tuple(inputs[name] for name in input_names)
        return precomputed[key]

    return generate_map(
        filename=filename,
        group=group,
        axes=axes,
        evaluate=precomputed_map,
        constants=constants,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Thermochemistry map functions
# ---------------------------------------------------------------------------

def rp1_lox_sp_map(pressure, entropy, mixture_ratio, guess_temperature):
    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    reactants = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=reactants,
        mode="sp",
        pressure=pressure,
        entropy=entropy,
        guess_temperature=guess_temperature,
    )

    return {
        "specific_heat_ratio": eq.gamma,
        "gas_constant": eq.gas_constant,
        "density": eq.density,
        "enthalpy": eq.enthalpy,
        "entropy": eq.entropy,
        "speed_of_sound": eq.speed_of_sound,
        "temperature": eq.temperature,
        "dynamic_viscosity": eq.dynamic_viscosity,
        "prandtl": eq.prandtl
    }


def rp1_lox_tp_map(pressure, temperature, mixture_ratio):
    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    reactants = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=reactants,
        mode="tp",
        pressure=pressure,
        temperature=temperature,
    )

    return {
        "specific_heat_ratio": eq.gamma,
        "gas_constant": eq.gas_constant,
        "density": eq.density,
        "enthalpy": eq.enthalpy,
        "entropy": eq.entropy,
        "speed_of_sound": eq.speed_of_sound,
        "temperature": eq.temperature,
        "dynamic_viscosity": eq.dynamic_viscosity,
        "prandtl": eq.prandtl
    }








# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Chamber / injector-face state
    # -----------------------------------------------------------------------

    mixture_ratio = State(2.3)
    chamber_pressure = State(400 * psia_to_pa)
    ambient_pressure = State(101325.0)

    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    Props = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio.value,
    )

    InjectorFace = Equilibrium(
        reactants=Props,
        mode="hp",
        pressure=chamber_pressure.value,
    )


    # -----------------------------------------------------------------------
    # Optional TP/SP map generation
    # -----------------------------------------------------------------------

    if make_map:
        if use_multiprocessing:
            map_generator = generate_map_mp
            map_options = {
                "workers": map_workers,
                "chunksize": map_chunksize,
            }
        else:
            map_generator = generate_map
            map_options = {}

        map_generator(
            filename=filename,
            group="products_tp",
            axes=[
                Axis.linear(
                    "pressure",
                    start=350 * psia_to_pa,
                    stop=430 * psia_to_pa,
                    count=41,
                    units="Pa",
                ),
                Axis.linear(
                    "temperature",
                    start=1500.0,
                    stop=3600.0,
                    count=85,
                    units="K",
                ),
                Axis.linear(
                    "mixture_ratio",
                    start=2.2,
                    stop=2.4,
                    count=5,
                    units="",
                ),
            ],
            evaluate=rp1_lox_tp_map,
            overwrite=True,
            raise_errors=True,
            **map_options,
        )

        map_generator(
            filename=filename,
            group="products_sp",
            axes=[
                Axis.log(
                    "pressure",
                    start=1 * psia_to_pa,
                    stop=500 * psia_to_pa,
                    count=18,
                    units="Pa",
                ),
                Axis.linear(
                    "entropy",
                    start=0.90 * InjectorFace.entropy,
                    stop=1.10 * InjectorFace.entropy,
                    count=10,
                    units="J/kg-K",
                ),
                Axis.linear(
                    "mixture_ratio",
                    start=1,
                    stop=4,
                    count=9,
                    units="",
                ),
            ],
            constants={
                "guess_temperature": InjectorFace.temperature,
            },
            evaluate=rp1_lox_sp_map,
            overwrite=True,
            raise_errors=True,
            **map_options,
        )















    # -----------------------------------------------------------------------
    # Heatsink Thrust Chamber Assembly
    # -----------------------------------------------------------------------

    Heatsink = Network("Heatsink")

    chamber_pressure = State(393 * psia_to_pa)
    chamber_temperature = State(3458.0)

    throat_pressure = State(300 * psia_to_pa)

    exit_pressure = State(10 * psia_to_pa)

    throat_area = 6 / 1550
    expansion_ratio = 6
    exit_area = throat_area * expansion_ratio

    chamber_diameter = 5.75 / 39.37
    chamber_area = (np.pi / 4) * chamber_diameter**2


    wall_thickness = 1 / 39.37
    element_length = 0.25 / 39.37
    steel = Material("1018")
    metal_volume = (np.pi/4) * ((6.75 / 39.37)**2 - (5.75 / 39.37)**2) * element_length
    metal_mass = metal_volume * steel.density
    hot_wall_area = element_length * np.pi * chamber_diameter
    cold_wall_area = element_length * np.pi * (6.75 / 39.37)


    ChamberMap = Map.from_hdf5(
        "Chamber Map",
        Heatsink,
        filename=filename,
        group="products_tp",
        inputs={
            "pressure": chamber_pressure,
            "temperature": chamber_temperature,
            "mixture_ratio": mixture_ratio,
        },
    )

    chamber_entropy = ChamberMap.entropy

    ThroatMap = Map.from_hdf5(
        "Throat Map",
        Heatsink,
        filename=filename,
        group="products_sp",
        inputs={
            "pressure": throat_pressure,
            "entropy": chamber_entropy,
            "mixture_ratio": mixture_ratio,
        },
    )

    ExitMap = Map.from_hdf5(
        "Exit Map",
        Heatsink,
        filename=filename,
        group="products_sp",
        inputs={
            "pressure": exit_pressure,
            "entropy": chamber_entropy,
            "mixture_ratio": mixture_ratio,
        },
    )


    Cyl = AdiabaticFlow(
        "Chamber Cylinder",
        Heatsink,
        upstream_static_enthalpy=InjectorFace.enthalpy,
        downstream_static_enthalpy=ChamberMap.enthalpy,
        downstream_density=ChamberMap.density,
        downstream_cross_sectional_area=chamber_area,
    )

    Chamber = Volume(
        "Combustion Chamber",
        Heatsink,
        pressure=chamber_pressure,
        temperature=chamber_temperature,
        enthalpy=ChamberMap.enthalpy,
        energy_variable="T",
        mass_flow_in=Cyl.mass_flow,
        total_enthalpy_in=InjectorFace.enthalpy,
    )

    Conv = AdiabaticFlow(
        "Converging Section",
        Heatsink,
        upstream_static_enthalpy=ChamberMap.enthalpy,
        upstream_density=ChamberMap.density,
        upstream_cross_sectional_area=chamber_area,
        downstream_static_enthalpy=ThroatMap.enthalpy,
        downstream_density=ThroatMap.density,
        downstream_cross_sectional_area=throat_area,
        mass_flow=Chamber.mass_flow_out,
        total_enthalpy=Chamber.total_enthalpy_out,
    )

    Throat = Volume(
        "Throat",
        Heatsink,
        pressure=throat_pressure,
        mass_flow_in=Conv.mass_flow,
    )

    Div = AdiabaticFlow(
        "Diverging Section",
        Heatsink,
        upstream_static_enthalpy=ThroatMap.enthalpy,
        downstream_static_enthalpy=ExitMap.enthalpy,
        upstream_density=ThroatMap.density,
        downstream_density=ExitMap.density,
        upstream_cross_sectional_area=throat_area,
        downstream_cross_sectional_area=exit_area,
        mass_flow=Throat.mass_flow_out,
    )

    chamber_velocity = Cyl.mass_flow / (ChamberMap.density * Cyl.downstream_cross_sectional_area)
    throat_velocity = Conv.mass_flow / (ThroatMap.density * throat_area)
    exit_velocity = Div.mass_flow / (ExitMap.density * exit_area)

    chamber_mach = abs(chamber_velocity) / ChamberMap.speed_of_sound
    throat_mach = abs(throat_velocity) / ThroatMap.speed_of_sound
    exit_mach = abs(exit_velocity) / ExitMap.speed_of_sound

    JetBoundary = Balance(
        "Jet Boundary",
        Heatsink,
        variable=exit_pressure,
        function=throat_mach - 1.0,
    )

    thrust = Conv.mass_flow * exit_velocity + (exit_pressure - ambient_pressure) * exit_area
    specific_impulse = thrust / (Conv.mass_flow * 9.80665)

    Heatsink.track("Injector Face Pressure [psia]", InjectorFace.pressure / psia_to_pa)
    Heatsink.track("Chamber Pressure [psia]", chamber_pressure / psia_to_pa)
    Heatsink.track("Throat Pressure [psia]", throat_pressure / psia_to_pa)
    Heatsink.track("Exit Pressure [psia]", exit_pressure / psia_to_pa)
    Heatsink.track("Ambient Pressure [psia]", ambient_pressure / psia_to_pa)

    Heatsink.track("Injector Face Temperature [K]", InjectorFace.temperature)
    Heatsink.track("Chamber Temperature [K]", chamber_temperature)
    Heatsink.track("Throat Temperature [K]", ThroatMap.temperature)
    Heatsink.track("Exit Temperature [K]", ExitMap.temperature)

    Heatsink.track("Chamber Gamma", ChamberMap.specific_heat_ratio)
    Heatsink.track("Throat Gamma", ThroatMap.specific_heat_ratio)
    Heatsink.track("Exit Gamma", ExitMap.specific_heat_ratio)

    Heatsink.track("Chamber Mach", chamber_mach)
    Heatsink.track("Throat Mach", throat_mach)
    Heatsink.track("Exit Mach", exit_mach)

    Heatsink.track("Cylinder Mass Flow [kg/s]", Cyl.mass_flow)
    Heatsink.track("Nozzle Mass Flow [kg/s]", Conv.mass_flow)

    Heatsink.track("Thrust [lbf]", thrust / 4.448)
    Heatsink.track("Specific Impulse [s]", specific_impulse)


    '''
    InjWallMetal = Lookup(
        "Injector Face Wall Metal",
        Heatsink,
        Material,
        "1018",
        temperature = 298.15
    )

    inj_wall_mean_temp = 0.5 * (InjectorFace.temperature + InjWallMetal.temperature)
    inj_pressure = State(InjectorFace.pressure)

    InjTamLookup = Map.from_hdf5(
        "Injector Face Mean Temp Lookup",
        Heatsink,
        filename=filename,
        group="products_tp",
        inputs={
            "pressure": inj_pressure,
            "temperature": inj_wall_mean_temp,
            "mixture_ratio": mixture_ratio,
        },
    )

    InjWall = Solid(
        "Injector Face Wall",
        Heatsink,
        temperature = InjWallMetal.temperature,
        mass=metal_mass,
        specific_heat=InjWallMetal.specific_heat,
    )

    InjBartz = Bartz(
        "Injector Face Wall Bartz",
        Heatsink,
        mass_flow=Conv.mass_flow,
        hydraulic_diameter=chamber_diameter,
        chamber_specific_heat_cp=InjectorFace.specific_heat_cp,
        chamber_prandtl_number=InjectorFace.prandtl,
        chamber_dynamic_viscosity=InjectorFace.dynamic_viscosity,
        local_freestream_density=InjectorFace.density,
        mean_temperature_density=InjTamLookup.density,
        mean_temperature_dynamic_viscosity=InjTamLookup.dynamic_viscosity,
    )

    InjRecoveryFactor = TemperatureRecoveryFactor(
        "Injector Face Temp Recovery Factor",
        Heatsink,
        prandtl_number=InjectorFace.prandtl
    )

    InjTaw = AdiabaticWallTemperature(
        "Injector face Adiabatic Wall Temp",
        Heatsink,
        total_temperature=InjectorFace.temperature,
        static_temperature=InjectorFace.temperature,
        recovery_factor=InjRecoveryFactor.recovery_factor
    )
    
    InjHotConvection = Convection(
        "Inj Hot Side Convection",
        Heatsink,
        surface_temperature=InjWall.temperature,
        fluid_temperature=InjTaw.adiabatic_wall_temperature,
        convective_area=hot_wall_area,
        convection_coefficient=InjBartz.convection_coefficient,
    )


    InjWall.heat_rate = InjHotConvection.heat_rate
    '''

    Transient(Heatsink).solve(
        dt = 0.01,
        t_final=1.0,
        verbose=True,
        statistics=True,
        filename=filename
    )
