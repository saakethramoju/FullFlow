"""
HDF5 Map example.

Large maps should usually be stored in an HDF5 file instead of written by hand
inside a script. FullFlow's Map.from_hdf5() does not require a FullPlot-specific
file. It only needs a simple rectangular-grid HDF5 layout:

    /<map_group>/axes/<axis_name>
    /<map_group>/outputs/<output_name>

This example uses FullPlot to generate that generic HDF5 layout, then loads the
file with Map.from_hdf5(). FullPlot is used here as a convenient map writer,
but Map.from_hdf5() only needs the generic HDF5 map layout shown above.
Compatible files can also be written by h5py or any other HDF5 tool.

The output datasets are renamed for use in the script:

    density  -> GasMap.rho
    enthalpy -> GasMap.h
"""

from fullflow import *
from fullplot import Axis, generate_map


map_filename = "9map_hdf5.h5"
map_group = "ideal_gas"


def ideal_gas_properties(pressure, temperature, gas_constant, specific_heat):
    return {
        "density": pressure / (gas_constant * temperature),
        "enthalpy": specific_heat * temperature,
        "gas_constant": gas_constant,
    }


# Generate the generic HDF5 map layout. The axis names become the input names
# used later by Map.from_hdf5(). Constants are passed to every evaluation point
# but are not swept as map axes.
generate_map(
    map_filename,
    group=map_group,
    axes=[
        Axis.linear("pressure", start=100000.0, stop=500000.0, count=5),
        Axis.linear("temperature", start=250.0, stop=500.0, count=6),
    ],
    constants={
        "gas_constant": 287.0,
        "specific_heat": 1005.0,
    },
    evaluate=ideal_gas_properties,
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
    group=map_group,
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
