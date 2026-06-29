from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from fullflow import *
from thermoprop import *


psia_to_pa = 6894.76

filename = "equilibrium_nozzle"

use_multiprocessing = True
map_workers = min(4, os.cpu_count() or 1)
map_chunksize = 25


# ---------------------------------------------------------------------------
# Map helper functions
# ---------------------------------------------------------------------------

def _inputs_from_index(axes, index, constants):
    inputs = dict(constants)

    for axis, i in zip(axes, index):
        inputs[axis.name] = float(axis.values[i])

    return inputs


def _map_key(inputs, input_names):
    return tuple(inputs[name] for name in input_names)


def _evaluate_map_point(job):
    evaluate, axes, constants, input_names, index = job

    inputs = _inputs_from_index(axes, index, constants)
    values = evaluate(**inputs)

    return _map_key(inputs, input_names), values


def generate_map_optional_multiprocessing(
    filename,
    axes,
    evaluate,
    group="map",
    outputs=None,
    constants=None,
    metadata=None,
    resume=True,
    overwrite=False,
    fill_value=np.nan,
    compression="gzip",
    compression_opts=None,
    flush_every=25,
    raise_errors=False,
    use_multiprocessing=False,
    workers=None,
    chunksize=25,
):
    """
    Generate a FullFlow map using normal generate_map(), with optional
    script-level multiprocessing.

    When use_multiprocessing=False, this is exactly a normal generate_map()
    call.

    When use_multiprocessing=True, worker processes evaluate the expensive
    map points first. The main process then writes the HDF5 file using the
    normal FullFlow generate_map() function. This avoids multiple processes
    writing to the same HDF5 file.

    Notes
    -----
    Cross-platform multiprocessing requires the call site to be inside:

        if __name__ == "__main__":

    This is required on Windows and macOS and is harmless on Linux.
    """

    axes = list(axes)
    constants = {} if constants is None else dict(constants)

    if not use_multiprocessing or workers is None or workers <= 1:
        return generate_map(
            filename=filename,
            group=group,
            axes=axes,
            evaluate=evaluate,
            outputs=outputs,
            constants=constants,
            metadata=metadata,
            resume=resume,
            overwrite=overwrite,
            fill_value=fill_value,
            compression=compression,
            compression_opts=compression_opts,
            flush_every=flush_every,
            raise_errors=raise_errors,
        )

    input_names = tuple(list(constants.keys()) + [axis.name for axis in axes])
    shape = tuple(len(axis.values) for axis in axes)
    indices = list(np.ndindex(shape))

    jobs = [
        (evaluate, axes, constants, input_names, index)
        for index in indices
    ]

    print(f"{group}: evaluating {len(jobs)} map points with {workers} processes...")

    precomputed = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for counter, (key, values) in enumerate(
            pool.map(_evaluate_map_point, jobs, chunksize=chunksize),
            start=1,
        ):
            precomputed[key] = values

            if counter % 100 == 0 or counter == len(jobs):
                print(f"{group}: completed {counter} / {len(jobs)}")

    def precomputed_map(**inputs):
        key = _map_key(inputs, input_names)
        return precomputed[key]

    return generate_map(
        filename=filename,
        group=group,
        axes=axes,
        evaluate=precomputed_map,
        outputs=outputs,
        constants=constants,
        metadata=metadata,
        resume=resume,
        overwrite=overwrite,
        fill_value=fill_value,
        compression=compression,
        compression_opts=compression_opts,
        flush_every=flush_every,
        raise_errors=raise_errors,
    )


# ---------------------------------------------------------------------------
# Thermochemistry map functions
# ---------------------------------------------------------------------------

def rp1_lox_tp_map(pressure, temperature, mixture_ratio):

    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    r = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=r,
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
    }


def rp1_lox_sp_map(pressure, entropy, mixture_ratio, guess_temperature):

    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    r = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=r,
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
        "temperature": eq.temperature
    }








# ---------------------------------------------------------------------------
# Map generation
#
# The __main__ guard is required for optional multiprocessing on Windows and
# macOS. It is harmless on Linux. If use_multiprocessing=False, the script still
# works normally.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mixture_ratio = State(2.3)
    chamber_pressure = State(400 * psia_to_pa)
    make_maps = False




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






    if make_maps:

        # Use the ideal chamber entropy only to choose a reasonable entropy-axis
        # range. The SP map itself is still a general SP(P, s, MR) map.
        reference_entropy = InjectorFace.entropy
        reference_temperature = InjectorFace.temperature

        generate_map_optional_multiprocessing(
            filename=filename,
            group="products_tp",
            axes=[
                Axis.log(
                    "pressure",
                    start=1 * psia_to_pa,
                    stop=500 * psia_to_pa,
                    count=16,
                    units="Pa",
                ),
                Axis.linear(
                    "temperature",
                    start=500,
                    stop=4500,
                    count=25,
                    units="K",
                ),
                Axis.linear(
                    "mixture_ratio",
                    start=1,
                    stop=4,
                    count=11,
                    units="",
                ),
            ],
            evaluate=rp1_lox_tp_map,
            overwrite=True,
            raise_errors=True,
            use_multiprocessing=use_multiprocessing,
            workers=map_workers,
            chunksize=map_chunksize,
        )

        generate_map_optional_multiprocessing(
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
                    start=0.90 * reference_entropy,
                    stop=1.10 * reference_entropy,
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
                "guess_temperature": reference_temperature,
            },
            evaluate=rp1_lox_sp_map,
            overwrite=True,
            raise_errors=True,
            use_multiprocessing=use_multiprocessing,
            workers=4,
            chunksize=10,
        )





    EquilibriumNozzle = Network("Equilibrium Nozzle")

    throat_pressure = State(chamber_pressure.value)
    throat_entropy = State(InjectorFace.entropy)


    ThroatMap = Map.from_hdf5(
        "Throat Map",
        EquilibriumNozzle,
        filename=filename,
        group="products_sp",
        inputs={
            "pressure": throat_pressure,
            "entropy": throat_entropy,
            "mixture_ratio": mixture_ratio
        }
    )


    Conv = CompressibleOrifice(
        "Converging Section",
        EquilibriumNozzle,
        upstream_total_pressure=InjectorFace.pressure,
        upstream_total_temperature=InjectorFace.temperature,
        downstream_pressure=throat_pressure,
        discharge_coefficient=1,
        cross_sectional_area=5.75/1550,
        gas_constant=InjectorFace.gas_constant,
        specific_heat_ratio=InjectorFace.gamma,
    )


    Throat = Volume(
        "Throat Node",
        EquilibriumNozzle,
        pressure=throat_pressure,
        mass_flow_in=Conv.mass_flow,
    )

    

    SteadyState(EquilibriumNozzle).solve(verbose=True)