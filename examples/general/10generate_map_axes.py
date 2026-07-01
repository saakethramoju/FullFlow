"""
generate_map Axis example.

A Map needs axes before it can be generated. An Axis is one independent input
that gets swept while the map file is built. The Axis name matters because it
becomes the keyword passed into the map function and the input name used later
by Map.from_hdf5().

This example shows the three common ways to make axes:

    Axis.linear()
        Even spacing between start and stop. Good for temperature, mixture
        ratio, valve position, or other variables that behave smoothly.

    Axis.log()
        Logarithmic spacing between start and stop. Good for pressure or any
        positive variable that spans a large range.

    Axis.values()
        Explicit user-provided breakpoints. Good for measured data, hand-picked
        points, pump speed lines, or any nonuniform grid.

The example generates a small ideal-gas map and then reads it back with
Map.from_hdf5().
"""

from fullflow import *


# ---------------------------------------------------------------------------
# Map filename
# ---------------------------------------------------------------------------

# The .h5 extension is added automatically if it is not included.
map_filename = "10generate_map_axes"


# ---------------------------------------------------------------------------
# Map function
# ---------------------------------------------------------------------------

# generate_map() calls this function at every combination of axis values.
#
# Axis names are passed as keyword arguments. Since the axes below are named
# pressure, temperature, and mixture_ratio, this function must accept those
# names.
def gas_properties(pressure, temperature, mixture_ratio, gas_constant, specific_heat):
    density = pressure / (gas_constant * temperature)
    enthalpy = specific_heat * temperature

    # The map function must return a flat dictionary of scalar numeric outputs.
    # Each key becomes one output dataset in the HDF5 map.
    return {
        "density": density,
        "enthalpy": enthalpy,
        "fuel_fraction": 1.0 / (1.0 + mixture_ratio),
        "oxidizer_fraction": mixture_ratio / (1.0 + mixture_ratio),
    }


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

# Linear axis: every temperature interval is the same size.
temperature_axis = Axis.linear(
    "temperature",
    start=250.0,
    stop=600.0,
    count=8,
    units="K",
)

# Log axis: pressure points are closer together on a logarithmic scale.
# The stored values are still real pressure values, not log(pressure).
pressure_axis = Axis.log(
    "pressure",
    start=100000.0,
    stop=1000000.0,
    count=7,
    units="Pa",
)

# Values axis: these are exactly the points the user chose.
mixture_ratio_axis = Axis.values(
    "mixture_ratio",
    values=[1.5, 2.0, 2.3, 2.6, 3.0],
    units="",
)


# ---------------------------------------------------------------------------
# Generate the map
# ---------------------------------------------------------------------------

# constants are also passed to gas_properties(), but they are not swept as map
# axes. Use constants for fixed inputs that should be stored with the map.
generate_map(
    filename=map_filename,
    group="axis_demo",
    axes=[
        pressure_axis,
        temperature_axis,
        mixture_ratio_axis,
    ],
    constants={
        "gas_constant": 287.0,
        "specific_heat": 1005.0,
    },
    evaluate=gas_properties,
    overwrite=True,
    raise_errors=True,
)


# ---------------------------------------------------------------------------
# Use the generated map
# ---------------------------------------------------------------------------

AxisMapNetwork = Network("Axis Map Example")

pressure = State(250000.0)
temperature = State(425.0)
mixture_ratio = State(2.3)

GasMap = Map.from_hdf5(
    "Gas Map",
    AxisMapNetwork,
    filename=map_filename,
    group="axis_demo",
    inputs={
        "pressure": pressure,
        "temperature": temperature,
        "mixture_ratio": mixture_ratio,
    },
)

GasMap.evaluate_states()

print("Pressure:", pressure.value)
print("Temperature:", temperature.value)
print("Mixture ratio:", mixture_ratio.value)
print("Density:", GasMap.density.value)
print("Enthalpy:", GasMap.enthalpy.value)
print("Fuel fraction:", GasMap.fuel_fraction.value)
print("Oxidizer fraction:", GasMap.oxidizer_fraction.value)


# ---------------------------------------------------------------------------
# Change inputs and evaluate again
# ---------------------------------------------------------------------------

# The map is connected to the input States. Change an input, evaluate again,
# and the output States update from the same HDF5 map.
pressure.value = 600000.0
temperature.value = 500.0
mixture_ratio.value = 2.6

GasMap.evaluate_states()

print("\nAfter changing the map inputs:")
print("Pressure:", pressure.value)
print("Temperature:", temperature.value)
print("Mixture ratio:", mixture_ratio.value)
print("Density:", GasMap.density.value)
print("Enthalpy:", GasMap.enthalpy.value)
