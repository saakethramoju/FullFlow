"""
Lookup class and chained lookup example.

Lookup can wrap a class just as easily as a function. This is useful when a
property model naturally stores many attributes on an object.

This example also shows lookup chaining. The Flow lookup receives density from
the Air lookup. When the Air state changes, the Flow result updates too.
"""

from fullflow import *


LookupNetwork = Network("Lookup Class and Chaining Example")


class SimpleAir:
    def __init__(self, pressure, temperature):
        self.pressure = pressure
        self.temperature = temperature
        self.gas_constant = 287.0
        self.specific_heat = 1005.0
        self.density = pressure / (self.gas_constant * temperature)
        self.enthalpy = self.specific_heat * temperature
        self.speed_of_sound = (1.4 * self.gas_constant * temperature) ** 0.5


def pipe_flow(mass_flow, density, area):
    velocity = mass_flow / (density * area)

    return {
        "velocity": velocity,
    }


pressure = State(101325.0)
temperature = State(300.0)
mass_flow = State(0.5)
pipe_area = State(0.01)

Air = Lookup(
    "Air",
    LookupNetwork,
    SimpleAir,
    pressure=pressure,
    temperature=temperature,
)

Flow = Lookup(
    "Pipe Flow",
    LookupNetwork,
    pipe_flow,
    mass_flow=mass_flow,
    density=Air.density,
    area=pipe_area,
)

LookupNetwork.track("Air Density [kg/m3]", Air.density)
LookupNetwork.track("Air Speed of Sound [m/s]", Air.speed_of_sound)
LookupNetwork.track("Pipe Velocity [m/s]", Flow.velocity)

print("Initial density:", Air.density.value)
print("Initial pipe velocity:", Flow.velocity.value)

# Change the temperature and read the outputs again. The Air lookup updates,
# and the Flow lookup sees the new density.
temperature.value = 350.0

print("\nAfter changing air temperature:")
print("Density:", Air.density.value)
print("Pipe velocity:", Flow.velocity.value)
