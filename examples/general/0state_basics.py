"""
State basics example.

A State is the basic value holder used throughout FullFlow. Most component
inputs and outputs are States. The reason FullFlow uses States instead of plain
floats is that States can stay connected while a solver changes their values.

This example shows the most common State behavior:

    1. Store a value in .value.
    2. Use States in normal math expressions.
    3. Let derived States update automatically.
    4. Use math helpers such as sqrt().

This file does not need a Network because it is only demonstrating State
objects by themselves.
"""

from fullflow import *


# ---------------------------------------------------------------------------
# Independent States
# ---------------------------------------------------------------------------

# These States hold ordinary numbers.
pressure = State(101325.0)          # Pa
temperature = State(300.0)          # K
gas_constant = State(287.0)         # J/kg-K
velocity = State(25.0)              # m/s

print("Initial pressure:", pressure.value)
print("Initial temperature:", temperature.value)


# ---------------------------------------------------------------------------
# Derived States
# ---------------------------------------------------------------------------

# State math creates another State. The new State is derived from the original
# States. It does not freeze the current answer.
density = pressure / (gas_constant * temperature)
dynamic_pressure = 0.5 * density * velocity**2
stagnation_pressure = pressure + dynamic_pressure

print("Density:", density.value)
print("Dynamic pressure:", dynamic_pressure.value)
print("Approximate stagnation pressure:", stagnation_pressure.value)


# ---------------------------------------------------------------------------
# Derived States update when an input State changes
# ---------------------------------------------------------------------------

# Changing velocity automatically changes dynamic_pressure and
# stagnation_pressure because they are derived States.
velocity.value = 50.0

print("\nAfter changing velocity:")
print("Velocity:", velocity.value)
print("Dynamic pressure:", dynamic_pressure.value)
print("Approximate stagnation pressure:", stagnation_pressure.value)


# ---------------------------------------------------------------------------
# Common math helpers
# ---------------------------------------------------------------------------

# Some math operations are provided as State methods. These also return derived
# States.
gamma = State(1.4)
speed_of_sound = (gamma * gas_constant * temperature).sqrt()
mach_number = velocity / speed_of_sound

print("\nSpeed of sound:", speed_of_sound.value)
print("Mach number:", mach_number.value)


# ---------------------------------------------------------------------------
# Deriving an existing State after it has already been created
# ---------------------------------------------------------------------------

# Sometimes it is useful to make a placeholder State first and connect it later.
# This is common when a State needs to be created before the equation that
# defines it is available.
pressure_ratio = State()

# The <<= operator means "make this State derive from this expression."
# It does not copy the current value once. Instead, it keeps pressure_ratio
# connected to the expression on the right side.
pressure_ratio <<= stagnation_pressure / pressure

print("\nPressure ratio:", pressure_ratio.value)

# Because pressure_ratio is now derived, changing pressure still updates it.
pressure.value = 150000.0
print("Pressure ratio after changing pressure:", pressure_ratio.value)

# You can think of this:
#
#     pressure_ratio <<= stagnation_pressure / pressure
#
# as the delayed version of this:
#
#     pressure_ratio = stagnation_pressure / pressure
#
# The difference is that <<= lets you keep the original State object and attach
# its equation later.
