"""
Lookup function example.

Lookup wraps a normal Python function, class, or external model so that it can
be used inside a FullFlow network.

This example wraps a simple ideal-gas function. The function returns a
dictionary. Each dictionary key becomes a Lookup output:

    Gas.density
    Gas.enthalpy

The input values can be normal numbers or States. When an input State changes,
the Lookup output updates the next time the output is read or evaluated.
"""

from fullflow import *


GasNetwork = Network("Lookup Function Example")


def ideal_gas(pressure, temperature, gas_constant, specific_heat):
    density = pressure / (gas_constant * temperature)
    enthalpy = specific_heat * temperature

    return {
        "density": density,
        "enthalpy": enthalpy,
    }


pressure = State(101325.0)
temperature = State(300.0)

Gas = Lookup(
    "Air",
    GasNetwork,
    ideal_gas,
    pressure=pressure,
    temperature=temperature,
    gas_constant=287.0,
    specific_heat=1005.0,
)

GasNetwork.track("Pressure [Pa]", Gas.pressure)
GasNetwork.track("Temperature [K]", Gas.temperature)
GasNetwork.track("Density [kg/m3]", Gas.density)
GasNetwork.track("Enthalpy [J/kg]", Gas.enthalpy)

print("Initial pressure:", Gas.pressure.value)
print("Initial temperature:", Gas.temperature.value)
print("Initial density:", Gas.density.value)
print("Initial enthalpy:", Gas.enthalpy.value)

# Change input States and read the Lookup outputs again.
pressure.value = 250000.0
temperature.value = 350.0

print("\nAfter changing pressure and temperature:")
print("Pressure:", Gas.pressure.value)
print("Temperature:", Gas.temperature.value)
print("Density:", Gas.density.value)
print("Enthalpy:", Gas.enthalpy.value)
