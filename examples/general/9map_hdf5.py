"""
HDF5 Map example.

Large maps should usually be stored in an HDF5 file instead of written by hand
inside a script. FullFlow uses generate_map() to create those files and
Map.from_hdf5() to read them back into a network.

This example creates a small ideal-gas property map with pressure and
temperature axes. Then it loads only two outputs and renames them for use in the
script:

    density  -> GasMap.rho
    enthalpy -> GasMap.h
"""

from fullflow import *


map_filename = "general_gas_map"


def ideal_gas_map(pressure, temperature, gas_constant, specific_heat):
    density = pressure / (gas_constant * temperature)
    enthalpy = specific_heat * temperature

    return {
        "density": density,
        "enthalpy": enthalpy,
        "gas_constant": gas_constant,
    }


# Generate a small map file. overwrite=True keeps this example repeatable.
generate_map(
    filename=map_filename,
    group="ideal_gas",
    axes=[
        Axis.linear("pressure", start=100000.0, stop=500000.0, count=5, units="Pa"),
        Axis.linear("temperature", start=250.0, stop=500.0, count=6, units="K"),
    ],
    constants={
        "gas_constant": 287.0,
        "specific_heat": 1005.0,
    },
    evaluate=ideal_gas_map,
    overwrite=True,
    raise_errors=True,
)

MapNetwork = Network("HDF5 Map Example")

pressure = State(250000.0)
temperature = State(325.0)

GasMap = Map.from_hdf5(
    "Gas Map",
    MapNetwork,
    filename=map_filename,
    group="ideal_gas",
    inputs={
        "pressure": pressure,
        "temperature": temperature,
    },
    outputs={
        "rho": "density",
        "h": "enthalpy",
    },
)

# This example does not need a solver. The map can be evaluated directly.
GasMap.evaluate_states()

print("Pressure:", pressure.value)
print("Temperature:", temperature.value)
print("Density from map:", GasMap.rho.value)
print("Enthalpy from map:", GasMap.h.value)

# Change an input and evaluate again.
pressure.value = 400000.0
GasMap.evaluate_states()

print("\nAfter changing pressure:")
print("Pressure:", pressure.value)
print("Density from map:", GasMap.rho.value)
