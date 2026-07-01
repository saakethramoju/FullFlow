"""
Finite-area combustor equilibrium nozzle example.

Physical layout
---------------

    RP-1 / LOX reactants
          |
          v

    +-------------------------------+
    |      Injector Face State      |
    |  HP equilibrium combustion    |
    |  Treated as stagnation-like   |
    |  zero-velocity state          |
    +-------------------------------+
          |
          |  Chamber Cylinder
          |  Adiabatic acceleration from injector-face h0
          |  to finite-area chamber-end velocity
          v

    +-------------------------------+
    |        Combustion Chamber     |
    |  TP(Pc, Tc, MR) map lookup    |
    |  finite-area chamber state    |
    |  chamber Mach > 0             |
    +-------------------------------+
          |
          |  Converging Section
          |  Adiabatic acceleration to throat
          v

    +-------------------------------+
    |            Throat             |
    |  SP(Pt, s_chamber, MR) map    |
    |  choked closure: Mt = 1       |
    +-------------------------------+
          |
          |  Diverging Section
          |  Adiabatic expansion
          v

    +-------------------------------+
    |          Exit Plane           |
    |  SP(Pe, s_chamber, MR) map    |
    |  supersonic no-shock branch   |
    +-------------------------------+
          |
          v

    Ambient pressure

Model description
-----------------
This example is a finite-area combustor model. The injector-face state is first
computed using HP equilibrium combustion. That state is treated like a
stagnation state. The chamber cylinder then converts part of the injector-face
enthalpy into finite-area chamber velocity, causing the chamber-end static
pressure to fall below the injector-face pressure.

The chamber state is looked up from a TP map. The throat and exit states are
looked up from an SP map using the chamber entropy, which represents an
isentropic, shock-free nozzle expansion from the finite-area chamber state.

The nozzle is choked for this chamber-pressure / ambient-pressure combination,
so the internal nozzle closure is:

    throat_mach = 1

Ambient pressure is not forced onto the internal exit pressure. It only appears
in the pressure-thrust term:

    F = mdot*ue + (Pe - Pamb)*Ae

Map generation note
-------------------
Generating the equilibrium TP and SP maps can take some time. This example
includes optional multiprocessing using ProcessPoolExecutor. Multiprocessing
can use several CPU cores, so it may require noticeable computing power while
the maps are being generated.

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

# All generated map groups are stored in this one HDF5 file.
filename = "equilibrium_nozzle_test"

# Set to True only when the maps need to be regenerated.
# Normal example runs should use the existing map file.
make_map = False

# Multiprocessing options for map generation.
# The map generation is the expensive part of this example.
use_multiprocessing = True
map_workers = min(4, os.cpu_count() or 1)
map_chunksize = 25


# ---------------------------------------------------------------------------
# Multiprocessing map helper
# ---------------------------------------------------------------------------

def _evaluate_map_job(job):
    """
    Evaluate one map point in a worker process.

    The job contains:
        evaluate:
            The thermochemistry function to call.

        axis_names / axis_values / index:
            The map-axis location for this point.

        constants:
            Extra constant inputs passed to every point.

        input_names:
            Input ordering used to create a consistent dictionary key.

    The worker returns:
        key:
            Tuple of input values.

        values:
            Dictionary of map outputs.
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
    Generate a FullFlow map with multiprocessing.

    Worker processes evaluate the expensive thermochemistry points first.
    The main process then writes the HDF5 map using FullFlow's normal
    generate_map() function.

    This avoids multiple worker processes writing to the same HDF5 file.
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

    This map is used for nozzle stations where the flow is assumed isentropic
    from the finite-area chamber state.

    Inputs
    ------
    pressure:
        Static pressure at the nozzle station.

    entropy:
        Chamber entropy used for the isentropic expansion.

    mixture_ratio:
        Oxidizer-to-fuel mass ratio.

    guess_temperature:
        Initial temperature guess for the SP equilibrium solve.

    Returns
    -------
    dict
        Thermodynamic properties needed by the nozzle model.
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
    }


def rp1_lox_tp_map(pressure, temperature, mixture_ratio):
    """
    RP-1 / LOX equilibrium products map using TP inputs.

    This map is used for the finite-area combustion chamber state, where the
    chamber pressure and chamber temperature are solved directly by the network.
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
    }


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Injector-face stagnation-like combustion state
    # -----------------------------------------------------------------------

    mixture_ratio = State(2.3)
    injector_face_pressure = State(400 * psia_to_pa)
    ambient_pressure = State(101325.0)

    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    Props = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio.value,
    )

    # HP equilibrium gives the injector-face combustion state.
    # In this finite-area combustor example, this is treated like a
    # stagnation state with negligible velocity.
    InjectorFace = Equilibrium(
        reactants=Props,
        mode="hp",
        pressure=injector_face_pressure.value,
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

        # Local TP map around the expected finite-area chamber state.
        # This map is intentionally narrow because the chamber correction is
        # small, so a broad coarse TP map can smear the enthalpy difference.
        map_generator(
            filename=filename,
            group="products_tp",
            axes=[
                Axis.linear(
                    "pressure",
                    start=385 * psia_to_pa,
                    stop=405 * psia_to_pa,
                    count=41,
                    units="Pa",
                ),
                Axis.linear(
                    "temperature",
                    start=3440,
                    stop=3470,
                    count=61,
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

        # SP map for the nozzle throat and exit.
        # The entropy axis is centered around the injector-face entropy so that
        # the chamber entropy and nozzle isentrope remain within the map.
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
    # Equilibrium nozzle network
    # -----------------------------------------------------------------------

    EquilibriumNozzle = Network("Equilibrium Nozzle")


    # -----------------------------------------------------------------------
    # Initial guesses for solved station states
    # -----------------------------------------------------------------------

    # Chamber pressure and temperature are solved by the combustion chamber
    # Volume. These are only initial guesses.
    chamber_pressure = State(393 * psia_to_pa)
    chamber_temperature = State(3458.0)

    # Throat and exit pressures are also solved by the network.
    throat_pressure = State(300 * psia_to_pa)
    exit_pressure = State(10 * psia_to_pa)


    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    # Chamber geometry.
    chamber_diameter = 5.75 / 39.37
    chamber_area = (np.pi / 4) * chamber_diameter**2

    # Nozzle geometry.
    throat_area = 6 / 1550
    expansion_ratio = 6
    exit_area = throat_area * expansion_ratio


    # -----------------------------------------------------------------------
    # Chamber TP map
    # -----------------------------------------------------------------------

    # The finite-area chamber state is a static state with nonzero velocity.
    # It is looked up from TP(Pchamber, Tchamber, MR).
    ChamberMap = Map.from_hdf5(
        "Chamber Map",
        EquilibriumNozzle,
        filename=filename,
        group="products_tp",
        inputs={
            "pressure": chamber_pressure,
            "temperature": chamber_temperature,
            "mixture_ratio": mixture_ratio,
        },
    )

    # The nozzle is treated as shock-free and isentropic from the finite-area
    # chamber state, so the throat and exit use this chamber entropy.
    chamber_entropy = ChamberMap.entropy


    # -----------------------------------------------------------------------
    # Nozzle SP maps
    # -----------------------------------------------------------------------

    ThroatMap = Map.from_hdf5(
        "Throat Map",
        EquilibriumNozzle,
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
        EquilibriumNozzle,
        filename=filename,
        group="products_sp",
        inputs={
            "pressure": exit_pressure,
            "entropy": chamber_entropy,
            "mixture_ratio": mixture_ratio,
        },
    )


    # -----------------------------------------------------------------------
    # Chamber cylinder: injector face to finite-area chamber state
    # -----------------------------------------------------------------------

    # The upstream density and upstream area are intentionally omitted.
    # That tells AdiabaticFlow to treat InjectorFace.enthalpy as total enthalpy:
    #
    #     h0 = h_injector
    #
    # Then the chamber cylinder solves:
    #
    #     h0 = h_chamber + 0.5*u_chamber^2
    #     mdot = rho_chamber*A_chamber*u_chamber
    #
    # This creates the finite-area combustor pressure drop from injector face
    # to combustion chamber end.
    Cyl = AdiabaticFlow(
        "Chamber Cylinder",
        EquilibriumNozzle,
        upstream_static_enthalpy=InjectorFace.enthalpy,
        downstream_static_enthalpy=ChamberMap.enthalpy,
        downstream_density=ChamberMap.density,
        downstream_cross_sectional_area=chamber_area,
    )


    # -----------------------------------------------------------------------
    # Combustion chamber volume
    # -----------------------------------------------------------------------

    # The chamber Volume solves chamber pressure and temperature by enforcing
    # mass and energy consistency between the chamber-cylinder inflow and the
    # nozzle outflow.
    #
    # The incoming total enthalpy is the injector-face stagnation enthalpy.
    # The outgoing total enthalpy becomes the upstream total enthalpy for the
    # converging nozzle section.
    Chamber = Volume(
        "Combustion Chamber",
        EquilibriumNozzle,
        pressure=chamber_pressure,
        temperature=chamber_temperature,
        enthalpy=ChamberMap.enthalpy,
        energy_variable="T",
        mass_flow_in=Cyl.mass_flow,
        total_enthalpy_in=InjectorFace.enthalpy,
    )


    # -----------------------------------------------------------------------
    # Converging section: chamber to throat
    # -----------------------------------------------------------------------

    # This section starts from the finite-area chamber state and accelerates to
    # the throat.
    #
    # Because upstream_density and upstream_cross_sectional_area are provided,
    # AdiabaticFlow treats the chamber as a finite-area static station:
    #
    #     h_chamber + 0.5*u_chamber^2 = h_throat + 0.5*u_throat^2
    #     rho_chamber*u_chamber*A_chamber = rho_throat*u_throat*A_throat
    Conv = AdiabaticFlow(
        "Converging Section",
        EquilibriumNozzle,
        upstream_static_enthalpy=ChamberMap.enthalpy,
        upstream_density=ChamberMap.density,
        upstream_cross_sectional_area=chamber_area,
        downstream_static_enthalpy=ThroatMap.enthalpy,
        downstream_density=ThroatMap.density,
        downstream_cross_sectional_area=throat_area,
        mass_flow=Chamber.mass_flow_out,
        total_enthalpy=Chamber.total_enthalpy_out,
    )


    # -----------------------------------------------------------------------
    # Throat node
    # -----------------------------------------------------------------------

    # This algebraic Volume supplies the throat outflow state. Its mass balance
    # forces the converging-section flow and diverging-section flow to match.
    Throat = Volume(
        "Throat",
        EquilibriumNozzle,
        pressure=throat_pressure,
        mass_flow_in=Conv.mass_flow,
    )


    # -----------------------------------------------------------------------
    # Diverging section: throat to exit
    # -----------------------------------------------------------------------

    # This section expands the choked throat flow to the exit plane.
    #
    # In this example, the low exit-pressure guess and the choked throat closure
    # select the supersonic, shock-free branch.
    Div = AdiabaticFlow(
        "Diverging Section",
        EquilibriumNozzle,
        upstream_static_enthalpy=ThroatMap.enthalpy,
        downstream_static_enthalpy=ExitMap.enthalpy,
        upstream_density=ThroatMap.density,
        downstream_density=ExitMap.density,
        upstream_cross_sectional_area=throat_area,
        downstream_cross_sectional_area=exit_area,
        mass_flow=Throat.mass_flow_out,
    )


    # -----------------------------------------------------------------------
    # Derived velocities and Mach numbers
    # -----------------------------------------------------------------------

    chamber_velocity = Cyl.mass_flow / (ChamberMap.density * Cyl.downstream_cross_sectional_area)
    throat_velocity = Conv.mass_flow / (ThroatMap.density * throat_area)
    exit_velocity = Div.mass_flow / (ExitMap.density * exit_area)

    chamber_mach = abs(chamber_velocity) / ChamberMap.speed_of_sound
    throat_mach = abs(throat_velocity) / ThroatMap.speed_of_sound
    exit_mach = abs(exit_velocity) / ExitMap.speed_of_sound


    # -----------------------------------------------------------------------
    # Jet boundary / choked nozzle closure
    # -----------------------------------------------------------------------

    # This chamber pressure and ambient pressure produce a choked nozzle.
    # Therefore, the internal nozzle closure is:
    #
    #     Mt = 1
    #
    # The exit pressure is solved as the shock-free isentropic exit pressure.
    # Ambient pressure is not imposed on the internal exit plane.
    JetBoundary = Balance(
        "Jet Boundary",
        EquilibriumNozzle,
        variable=exit_pressure,
        function=throat_mach - 1.0,
    )


    # -----------------------------------------------------------------------
    # Thrust and specific impulse
    # -----------------------------------------------------------------------

    # One-dimensional thrust:
    #
    #     F = mdot*ue + (Pe - Pamb)*Ae
    #
    # The pressure term is negative if the nozzle is overexpanded relative to
    # ambient, and positive if underexpanded.
    thrust = Conv.mass_flow * exit_velocity + (exit_pressure - ambient_pressure) * exit_area

    # Specific impulse:
    #
    #     Isp = F/(mdot*g0)
    specific_impulse = thrust / (Conv.mass_flow * 9.80665)


    # -----------------------------------------------------------------------
    # Tracked outputs
    # -----------------------------------------------------------------------

    EquilibriumNozzle.track("Injector Face Pressure [psia]", InjectorFace.pressure / psia_to_pa)
    EquilibriumNozzle.track("Chamber Pressure [psia]", chamber_pressure / psia_to_pa)
    EquilibriumNozzle.track("Throat Pressure [psia]", throat_pressure / psia_to_pa)
    EquilibriumNozzle.track("Exit Pressure [psia]", exit_pressure / psia_to_pa)
    EquilibriumNozzle.track("Ambient Pressure [psia]", ambient_pressure / psia_to_pa)

    EquilibriumNozzle.track("Injector Face Temperature [K]", InjectorFace.temperature)
    EquilibriumNozzle.track("Chamber Temperature [K]", chamber_temperature)
    EquilibriumNozzle.track("Throat Temperature [K]", ThroatMap.temperature)
    EquilibriumNozzle.track("Exit Temperature [K]", ExitMap.temperature)

    EquilibriumNozzle.track("Chamber Gamma", ChamberMap.specific_heat_ratio)
    EquilibriumNozzle.track("Throat Gamma", ThroatMap.specific_heat_ratio)
    EquilibriumNozzle.track("Exit Gamma", ExitMap.specific_heat_ratio)

    EquilibriumNozzle.track("Chamber Mach", chamber_mach)
    EquilibriumNozzle.track("Throat Mach", throat_mach)
    EquilibriumNozzle.track("Exit Mach", exit_mach)

    EquilibriumNozzle.track("Cylinder Mass Flow [kg/s]", Cyl.mass_flow)
    EquilibriumNozzle.track("Nozzle Mass Flow [kg/s]", Conv.mass_flow)

    EquilibriumNozzle.track("Thrust [lbf]", thrust / 4.448)
    EquilibriumNozzle.track("Specific Impulse [s]", specific_impulse)


    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------

    SteadyState(EquilibriumNozzle).solve(verbose=True)


    # -----------------------------------------------------------------------
    # Finite-area combustor pressure-ratio checks
    # -----------------------------------------------------------------------

    # These printed ratios are simple checks on the finite-area combustor
    # pressure drop. They are not enforced as balances in this model.
    contraction_ratio = chamber_area / throat_area

    ratio_predicted = InjectorFace.pressure / Chamber.pressure # value predicted by FullFlow model
    ratio_rayleigh = 1 + InjectorFace.specific_heat_ratio * chamber_mach**2 # value predicted by Rayleigh Flow equation 1 + gamma * M^2
    ratio_empirical = 1 + 0.54 / contraction_ratio**2.2 # empirical Rayleigh Line Loss equation

    print(f"Predicted Line Loss: {ratio_predicted.value:.3f}")
    print(f"Rayleigh Line Loss: {ratio_rayleigh.value:.3f}")
    print(f"Empirical Loss: {ratio_empirical:.3f}")