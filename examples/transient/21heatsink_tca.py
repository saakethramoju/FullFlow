"""
Heatsink thrust chamber assembly transient example.

Physical layout
---------------

    RP-1 / LOX reactants
          |
          v

    +------------------------------------------------+
    |                  Chamber State                 |
    |  HP equilibrium combustion at chamber pressure |
    |  Used as the chamber / injector-face gas state |
    +------------------------------------------------+
          |
          |  Converging nozzle section
          v

    +------------------------------------------------+
    |                    Throat                      |
    |  SP(Pt, s_chamber, MR) equilibrium lookup      |
    |  Exit pressure is adjusted until Mt = 1        |
    +------------------------------------------------+
          |
          |  Diverging nozzle section
          v

    +------------------------------------------------+
    |                   Exit Plane                   |
    |  SP(Pe, s_chamber, MR) equilibrium lookup      |
    |  Shock-free isentropic expansion branch        |
    +------------------------------------------------+
          |
          v

    Ambient pressure


Thermal model
-------------

A short axial slice of the chamber, throat, and exit wall is modeled. Each
station uses the same radial 3-node heatsink wall model:

        hot gas
          |
          |  Bartz hot-side convection
          v
    +-------------+
    | Wall Node 1 |  inner wall metal node
    +-------------+
          |
          |  radial conduction
          v
    +-------------+
    | Wall Node 2 |  middle wall metal node
    +-------------+
          |
          |  radial conduction
          v
    +-------------+
    | Wall Node 3 |  outer wall metal node
    +-------------+
          |
          |  natural convection to ambient air
          v
        ambient air

The hot-side heat rate is multiplied by a sooting factor. This represents the
reduction in heat transfer caused by carbon deposition on the hot-gas wall.

The wall nodes are real transient states. There is no steady thermal solution
at the start because the wall begins cold and heats up from the combustion gas.

Thermochemistry maps
--------------------

HP equilibrium:
    Used once to compute the chamber combustion state.

SP map:
    Used for throat and exit gas properties during the isentropic nozzle
    expansion.

TP map:
    Used only for heat-transfer property correction. Bartz needs gas density
    and viscosity at a gas/wall mean temperature, so the TP map is used as a
    film-property lookup.

Map generation note
-------------------

Generating the equilibrium maps can take time. This example includes optional
multiprocessing using ProcessPoolExecutor. Multiprocessing can use several CPU
cores, so it may require noticeable computing power while the maps are being
generated.

For normal runs, leave:

    make_map = False

To regenerate the maps, set:

    make_map = True

The if __name__ == "__main__" guard is required for multiprocessing to work
reliably on Windows and macOS, and it is also safe on Linux.
"""


from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Constants and script options
# ---------------------------------------------------------------------------

psia_to_pa = 6894.76

# All generated map groups are stored in this HDF5 file.
filename = "21heatsink_tca"

# Set this to True only when the maps need to be regenerated.
make_map = False

# Multiprocessing options used only during map generation.
use_multiprocessing = True
map_workers = min(4, os.cpu_count() or 1)
map_chunksize = 25


# ---------------------------------------------------------------------------
# Multiprocessing map helper
# ---------------------------------------------------------------------------

def _evaluate_map_job(job):
    """
    Evaluate one thermochemistry map point in a worker process.

    The worker receives the map function, the axis values, any constant inputs,
    and the multi-dimensional index for the point it should evaluate.

    It returns a key and the computed property dictionary. The main process
    stores those values and writes the HDF5 map.
    """

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
    """
    Generate a FullFlow map using multiprocessing.

    The expensive thermochemistry evaluations are done in worker processes.
    The HDF5 file is written only in the main process. This avoids multiple
    processes writing to the same HDF5 file at the same time.
    """

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
    """
    RP-1 / LOX equilibrium products map using SP inputs.

    This map is used at nozzle stations where the flow is assumed to expand
    isentropically from the chamber state.

    Inputs
    ------
    pressure:
        Static pressure at the nozzle station.

    entropy:
        Chamber entropy. Holding this fixed creates the isentropic nozzle
        expansion path.

    mixture_ratio:
        Oxidizer-to-fuel mass ratio.

    guess_temperature:
        Initial temperature guess for the SP equilibrium solve.

    Returns
    -------
    dict
        Gas properties needed by the nozzle and heat-transfer model.
    """

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
        "prandtl": eq.prandtl,
    }


def rp1_lox_tp_map(pressure, temperature, mixture_ratio):
    """
    RP-1 / LOX equilibrium products map using TP inputs.

    This map is used as a heat-transfer property map. Bartz uses gas density
    and viscosity at a gas/wall mean temperature, so pressure and temperature
    are the natural lookup inputs here.
    """

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
        "prandtl": eq.prandtl,
    }


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Chamber / injector-face combustion state
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

    # The chamber is computed with HP equilibrium combustion.
    #
    # This example no longer includes a separate finite-area chamber node.
    # The HP chamber state is used directly as the chamber gas state entering
    # the converging nozzle section.
    Chamber = Equilibrium(
        reactants=Props,
        mode="hp",
        pressure=chamber_pressure.value,
    )


    # -----------------------------------------------------------------------
    # Optional map generation
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

        # TP map used for heat-transfer film-property lookups.
        #
        # This map must cover the chamber, throat, and exit pressures, and it
        # must cover the gas/wall mean temperatures that occur while the wall
        # heats up.
        map_generator(
            filename=filename,
            group="products_tp",
            axes=[
                Axis.log(
                    "pressure",
                    start=1 * psia_to_pa,
                    stop=430 * psia_to_pa,
                    count=50,
                    units="Pa",
                ),
                Axis.linear(
                    "temperature",
                    start=300.0,
                    stop=3600.0,
                    count=90,
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

        # SP map used for the nozzle expansion.
        #
        # The entropy axis is centered around the chamber entropy. The throat
        # and exit lookups use this entropy to represent an isentropic nozzle.
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
                    start=0.90 * Chamber.entropy,
                    stop=1.10 * Chamber.entropy,
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
                "guess_temperature": Chamber.temperature,
            },
            evaluate=rp1_lox_sp_map,
            overwrite=True,
            raise_errors=True,
            **map_options,
        )


    # -----------------------------------------------------------------------
    # Network
    # -----------------------------------------------------------------------

    Heatsink = Network("Heatsink")


    # -----------------------------------------------------------------------
    # Nozzle solve variables and geometry
    # -----------------------------------------------------------------------

    # These are initial guesses for the algebraic nozzle solve.
    throat_pressure = State(229.427 * psia_to_pa)
    exit_pressure = State(10.4529 * psia_to_pa)

    # Throat area and exit area.
    throat_area = 6 / 1550
    expansion_ratio = 6
    exit_area = throat_area * expansion_ratio

    # Chamber area. The chamber state is used as the upstream state for the
    # converging nozzle section.
    chamber_diameter = 5.75 / 39.37
    chamber_area = (np.pi / 4) * chamber_diameter**2

    # Equivalent circular diameters used by the heat-transfer wall stations.
    throat_diameter = (4 * throat_area / np.pi)**0.5
    exit_diameter = (4 * exit_area / np.pi)**0.5


    # -----------------------------------------------------------------------
    # Wall geometry and thermal options
    # -----------------------------------------------------------------------

    # Each wall station uses three radial thermal nodes.
    wall_thickness = 0.9 / 39.37
    node_thickness = wall_thickness / 3
    element_length = 0.25 / 39.37

    steel = Material("1018")

    # Empirical multiplier applied to the hot-gas heat rate.
    # Values below 1 reduce the gas-side heat transfer.
    sooting_factor = 0.75

    # Ambient air temperature used by the outer-wall natural convection model.
    ambient_air_temperature = State(298.15)


    # -----------------------------------------------------------------------
    # Nozzle thermodynamic maps
    # -----------------------------------------------------------------------

    chamber_entropy = State(Chamber.entropy)

    # Throat state from the isentropic SP map.
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

    # Exit state from the same chamber isentrope.
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


    # -----------------------------------------------------------------------
    # Gas path: chamber to throat
    # -----------------------------------------------------------------------

    # The converging section accelerates from the chamber state to the throat.
    #
    # The component uses static enthalpy, density, and area at both stations.
    # It computes mass flow and total enthalpy from the energy relation:
    #
    #     h1 + u1^2/2 = h2 + u2^2/2
    #
    # and continuity:
    #
    #     mdot = rho*u*A
    Conv = AdiabaticFlow(
        "Converging Section",
        Heatsink,
        upstream_static_enthalpy=Chamber.enthalpy,
        upstream_density=Chamber.density,
        upstream_cross_sectional_area=chamber_area,
        downstream_static_enthalpy=ThroatMap.enthalpy,
        downstream_density=ThroatMap.density,
        downstream_cross_sectional_area=throat_area,
    )

    # Algebraic throat volume used to connect the converging and diverging
    # sections. Its mass balance forces the two section mass flows to match.
    Throat = Volume(
        "Throat",
        Heatsink,
        pressure=throat_pressure,
        mass_flow_in=Conv.mass_flow,
    )


    # -----------------------------------------------------------------------
    # Gas path: throat to exit
    # -----------------------------------------------------------------------

    # The diverging section expands the choked throat flow to the exit state.
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


    # -----------------------------------------------------------------------
    # Derived nozzle velocities and Mach numbers
    # -----------------------------------------------------------------------

    chamber_velocity = Conv.mass_flow / (Chamber.density * chamber_area)
    throat_velocity = Conv.mass_flow / (ThroatMap.density * throat_area)
    exit_velocity = Div.mass_flow / (ExitMap.density * exit_area)

    chamber_mach = abs(chamber_velocity) / Chamber.speed_of_sound
    throat_mach = abs(throat_velocity) / ThroatMap.speed_of_sound
    exit_mach = abs(exit_velocity) / ExitMap.speed_of_sound


    # -----------------------------------------------------------------------
    # Choked nozzle closure
    # -----------------------------------------------------------------------

    # The exit pressure is adjusted until the throat Mach number is one.
    # Ambient pressure is not imposed as the internal exit pressure.
    JetBoundary = Balance(
        "Jet Boundary",
        Heatsink,
        variable=exit_pressure,
        function=throat_mach - 1.0,
    )


    # -----------------------------------------------------------------------
    # Thrust and specific impulse
    # -----------------------------------------------------------------------

    thrust = Conv.mass_flow * exit_velocity + (exit_pressure - ambient_pressure) * exit_area
    specific_impulse = thrust / (Conv.mass_flow * 9.80665)


    # -----------------------------------------------------------------------
    # Nozzle tracked outputs
    # -----------------------------------------------------------------------

    Heatsink.track("Chamber Pressure [psia]", Chamber.pressure / psia_to_pa)
    Heatsink.track("Throat Pressure [psia]", throat_pressure / psia_to_pa)
    Heatsink.track("Exit Pressure [psia]", exit_pressure / psia_to_pa)
    Heatsink.track("Ambient Pressure [psia]", ambient_pressure / psia_to_pa)

    Heatsink.track("Chamber Temperature [K]", Chamber.temperature)
    Heatsink.track("Throat Temperature [K]", ThroatMap.temperature)
    Heatsink.track("Exit Temperature [K]", ExitMap.temperature)

    Heatsink.track("Chamber Gamma", Chamber.gamma)
    Heatsink.track("Throat Gamma", ThroatMap.specific_heat_ratio)
    Heatsink.track("Exit Gamma", ExitMap.specific_heat_ratio)

    Heatsink.track("Chamber Mach", chamber_mach)
    Heatsink.track("Throat Mach", throat_mach)
    Heatsink.track("Exit Mach", exit_mach)

    Heatsink.track("Nozzle Mass Flow [kg/s]", Conv.mass_flow)

    Heatsink.track("Thrust [lbf]", thrust / 4.448)
    Heatsink.track("Specific Impulse [s]", specific_impulse)


    # -----------------------------------------------------------------------
    # Wall helper functions
    # -----------------------------------------------------------------------

    def harmonic_mean(a, b):
        """
        Harmonic mean used for conduction between two metal nodes.

        This is useful when the adjacent wall nodes have different thermal
        conductivities because each half-cell contributes a thermal resistance.
        """

        return 2.0 * a * b / (a + b)


    def make_wall_geometry(inner_diameter):
        """
        Build the three-node radial wall geometry for one axial station.

        The wall is divided into three equal radial nodes. Because diameter is
        twice radius, each radial node increases diameter by:

            2 * node_thickness

        The conduction areas are evaluated at the interfaces between nodes.
        """

        diameter_inner = inner_diameter
        diameter_12 = inner_diameter + 2 * node_thickness
        diameter_23 = inner_diameter + 4 * node_thickness
        diameter_outer = inner_diameter + 6 * node_thickness

        metal_volume1 = (np.pi / 4) * (diameter_12**2 - diameter_inner**2) * element_length
        metal_volume2 = (np.pi / 4) * (diameter_23**2 - diameter_12**2) * element_length
        metal_volume3 = (np.pi / 4) * (diameter_outer**2 - diameter_23**2) * element_length

        return {
            "mass1": metal_volume1 * steel.density,
            "mass2": metal_volume2 * steel.density,
            "mass3": metal_volume3 * steel.density,
            "hot_area": element_length * np.pi * diameter_inner,
            "area12": element_length * np.pi * diameter_12,
            "area23": element_length * np.pi * diameter_23,
            "cold_area": element_length * np.pi * diameter_outer,
        }


    def make_wall_station(
        station_name,
        gas_pressure,
        gas_static_temperature,
        gas_total_temperature,
        gas_density,
        gas_prandtl,
        inner_diameter,
    ):
        """
        Build one complete three-node wall station.

        The station contains:

            1. Three temperature-dependent 1018 steel material lookups.
            2. Three Solid nodes for the radial thermal mass.
            3. Two radial Conduction links between the solid nodes.
            4. A TP equilibrium lookup for gas film properties.
            5. A Bartz hot-side gas convection coefficient.
            6. A recovery factor and adiabatic wall temperature.
            7. A hot-side convection heat rate.
            8. An outer-wall air-property lookup.
            9. A natural-convection air-side coefficient.
           10. An outer-wall convection heat rate.

        Positive heat rate means heat enters the solid node. The conduction and
        convection sign conventions are connected so energy is conserved between
        the wall nodes.
        """

        wall_geometry = make_wall_geometry(inner_diameter)

        # Each wall node starts at room temperature.
        wall_temperature1 = State(298.15)
        wall_temperature2 = State(298.15)
        wall_temperature3 = State(298.15)


        # -------------------------------------------------------------------
        # Temperature-dependent wall material properties
        # -------------------------------------------------------------------

        WallMetal1 = Lookup(
            f"{station_name} Wall Metal 1",
            Heatsink,
            Material,
            "1018",
            temperature=wall_temperature1,
        )

        WallMetal2 = Lookup(
            f"{station_name} Wall Metal 2",
            Heatsink,
            Material,
            "1018",
            temperature=wall_temperature2,
        )

        WallMetal3 = Lookup(
            f"{station_name} Wall Metal 3",
            Heatsink,
            Material,
            "1018",
            temperature=wall_temperature3,
        )


        # -------------------------------------------------------------------
        # Hot-gas film-property lookup
        # -------------------------------------------------------------------

        # Bartz uses gas density and viscosity evaluated at a gas/wall mean
        # temperature. The TP map is used here because the desired lookup state
        # is pressure + mean temperature, not pressure + entropy.
        gas_wall_mean_temperature = 0.5 * (gas_static_temperature + WallMetal1.temperature)

        WallTamLookup = Map.from_hdf5(
            f"{station_name} Mean Temp Lookup",
            Heatsink,
            filename=filename,
            group="products_tp",
            inputs={
                "pressure": gas_pressure,
                "temperature": gas_wall_mean_temperature,
                "mixture_ratio": mixture_ratio,
            },
        )


        # -------------------------------------------------------------------
        # Three radial wall thermal masses
        # -------------------------------------------------------------------

        WallNode1 = Solid(
            f"{station_name} Wall Node 1",
            Heatsink,
            temperature=wall_temperature1,
            mass=wall_geometry["mass1"],
            specific_heat=WallMetal1.specific_heat,
        )

        WallNode2 = Solid(
            f"{station_name} Wall Node 2",
            Heatsink,
            temperature=wall_temperature2,
            mass=wall_geometry["mass2"],
            specific_heat=WallMetal2.specific_heat,
        )

        WallNode3 = Solid(
            f"{station_name} Wall Node 3",
            Heatsink,
            temperature=wall_temperature3,
            mass=wall_geometry["mass3"],
            specific_heat=WallMetal3.specific_heat,
        )


        # -------------------------------------------------------------------
        # Radial conduction between wall nodes
        # -------------------------------------------------------------------

        Wall12Conduction = Conduction(
            f"{station_name} Node 1 to 2 Conduction",
            Heatsink,
            temperature1=WallNode1.temperature,
            temperature2=WallNode2.temperature,
            thermal_conductivity=harmonic_mean(WallMetal1.thermal_conductivity, WallMetal2.thermal_conductivity),
            length=node_thickness,
            conductive_area=wall_geometry["area12"],
        )

        Wall23Conduction = Conduction(
            f"{station_name} Node 2 to 3 Conduction",
            Heatsink,
            temperature1=WallNode2.temperature,
            temperature2=WallNode3.temperature,
            thermal_conductivity=harmonic_mean(WallMetal2.thermal_conductivity, WallMetal3.thermal_conductivity),
            length=node_thickness,
            conductive_area=wall_geometry["area23"],
        )


        # -------------------------------------------------------------------
        # Hot-side Bartz convection
        # -------------------------------------------------------------------

        # The freestream gas properties come from the station gas state. The
        # mean-temperature density and viscosity come from the TP film lookup.
        WallBartz = Bartz(
            f"{station_name} Wall Bartz",
            Heatsink,
            mass_flow=Conv.mass_flow,
            hydraulic_diameter=inner_diameter,
            chamber_specific_heat_cp=Chamber.specific_heat_cp,
            chamber_prandtl_number=Chamber.prandtl,
            chamber_dynamic_viscosity=Chamber.dynamic_viscosity,
            local_freestream_density=gas_density,
            mean_temperature_density=WallTamLookup.density,
            mean_temperature_dynamic_viscosity=WallTamLookup.dynamic_viscosity,
        )

        WallRecoveryFactor = TemperatureRecoveryFactor(
            f"{station_name} Temp Recovery Factor",
            Heatsink,
            prandtl_number=gas_prandtl,
        )

        WallTaw = AdiabaticWallTemperature(
            f"{station_name} Adiabatic Wall Temp",
            Heatsink,
            total_temperature=gas_total_temperature,
            static_temperature=gas_static_temperature,
            recovery_factor=WallRecoveryFactor.recovery_factor,
        )

        WallHotConvection = Convection(
            f"{station_name} Hot Side Convection",
            Heatsink,
            surface_temperature=WallNode1.temperature,
            fluid_temperature=WallTaw.adiabatic_wall_temperature,
            convective_area=wall_geometry["hot_area"],
            convection_coefficient=WallBartz.convection_coefficient,
        )


        # -------------------------------------------------------------------
        # Outer-wall natural convection to air
        # -------------------------------------------------------------------

        # Air properties are evaluated at the outside-wall film temperature.
        # The convection heat rate itself still uses ambient air temperature.
        air_film_temperature = 0.5 * (WallNode3.temperature + ambient_air_temperature)

        AirLookup = Lookup(
            f"{station_name} Outer Wall Air",
            Heatsink,
            Fluid,
            "air",
            pressure=101325.0,
            temperature=air_film_temperature,
        )

        AirNaturalConvection = NaturalConvection(
            f"{station_name} Outer Wall Air Natural Convection",
            Heatsink,
            wall_temperature=WallNode3.temperature,
            fluid_temperature=ambient_air_temperature,
            characteristic_length=element_length,
            fluid_density=AirLookup.density,
            fluid_specific_heat=AirLookup.specific_heat_cp,
            fluid_dynamic_viscosity=AirLookup.dynamic_viscosity,
            fluid_conductivity=AirLookup.conductivity,
            thermal_expansion_coefficient=1.0 / AirLookup.temperature,
        )

        AirConvection = Convection(
            f"{station_name} Outer Wall Air Convection",
            Heatsink,
            surface_temperature=WallNode3.temperature,
            fluid_temperature=ambient_air_temperature,
            convective_area=wall_geometry["cold_area"],
            convection_coefficient=AirNaturalConvection.convection_coefficient,
        )


        # -------------------------------------------------------------------
        # Solid-node energy connections
        # -------------------------------------------------------------------

        # Convection uses:
        #
        #     q = h*A*(Tfluid - Tsurface)
        #
        # so hot-side convection is positive when the gas is hotter than the
        # wall. Outer-wall convection is negative when the wall is hotter than
        # ambient air.
        #
        # Conduction uses:
        #
        #     q12 = k*A/L*(T2 - T1)
        #
        # so q12 is negative when node 1 is hotter than node 2.
        WallNode1.heat_rate = sooting_factor * WallHotConvection.heat_rate + Wall12Conduction.heat_rate
        WallNode2.heat_rate = -Wall12Conduction.heat_rate + Wall23Conduction.heat_rate
        WallNode3.heat_rate = -Wall23Conduction.heat_rate + AirConvection.heat_rate


        # -------------------------------------------------------------------
        # Wall tracked outputs
        # -------------------------------------------------------------------

        Heatsink.track(f"{station_name} Wall Node 1 Temperature [K]", WallNode1.temperature)
        Heatsink.track(f"{station_name} Wall Node 2 Temperature [K]", WallNode2.temperature)
        Heatsink.track(f"{station_name} Wall Node 3 Temperature [K]", WallNode3.temperature)

        Heatsink.track(f"{station_name} Hot Side Heat Rate [W]", WallHotConvection.heat_rate)
        Heatsink.track(f"{station_name} Node 1 to 2 Heat Rate [W]", Wall12Conduction.heat_rate)
        Heatsink.track(f"{station_name} Node 2 to 3 Heat Rate [W]", Wall23Conduction.heat_rate)
        Heatsink.track(f"{station_name} Outer Wall Air Heat Rate [W]", AirConvection.heat_rate)
        Heatsink.track(f"{station_name} Outer Wall Air Convection Coefficient [W/m2-K]", AirConvection.convection_coefficient)
        Heatsink.track(f"{station_name} Bartz Convective Coefficient [W/m2-K]", WallBartz.convection_coefficient)

        return {
            "node1": WallNode1,
            "node2": WallNode2,
            "node3": WallNode3,
            "hot_convection": WallHotConvection,
            "conduction12": Wall12Conduction,
            "conduction23": Wall23Conduction,
            "air_convection": AirConvection,
            "bartz": WallBartz,
        }


    # -----------------------------------------------------------------------
    # Wall stations
    # -----------------------------------------------------------------------

    # Chamber wall station.
    #
    # The chamber gas state is the HP equilibrium chamber state. Since this is
    # treated as the upstream chamber state, static and total temperature are
    # both passed as Chamber.temperature.
    ChamberWall = make_wall_station(
        station_name="Chamber",
        gas_pressure=chamber_pressure,
        gas_static_temperature=Chamber.temperature,
        gas_total_temperature=Chamber.temperature,
        gas_density=Chamber.density,
        gas_prandtl=Chamber.prandtl,
        inner_diameter=chamber_diameter,
    )

    # Throat wall station.
    #
    # The throat static state comes from the SP map. The total temperature is
    # approximated using the chamber temperature because the gas path is modeled
    # as adiabatic from the chamber to the throat.
    ThroatWall = make_wall_station(
        station_name="Throat",
        gas_pressure=throat_pressure,
        gas_static_temperature=ThroatMap.temperature,
        gas_total_temperature=Chamber.temperature,
        gas_density=ThroatMap.density,
        gas_prandtl=ThroatMap.prandtl,
        inner_diameter=throat_diameter,
    )

    # Exit wall station.
    #
    # The exit static state comes from the SP map. The total temperature is
    # again approximated using the chamber temperature.
    ExitWall = make_wall_station(
        station_name="Exit",
        gas_pressure=exit_pressure,
        gas_static_temperature=ExitMap.temperature,
        gas_total_temperature=Chamber.temperature,
        gas_density=ExitMap.density,
        gas_prandtl=ExitMap.prandtl,
        inner_diameter=exit_diameter,
    )


    # -----------------------------------------------------------------------
    # Transient solve
    # -----------------------------------------------------------------------

    Transient(Heatsink).solve(
        dt=0.1,
        t_final=10.0,
        verbose=True,
        statistics=True,
        filename=filename,
    )