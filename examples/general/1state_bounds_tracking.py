"""
State bounds, containers, and tracking example.

This example shows three useful State features that appear often in user
models:

    1. Bounds for solver variables.
    2. State.from_iterable() for dictionaries and lists.
    3. Network.track() for values that should appear in solver output files.

Bounds are useful when a solver variable should never become negative or should
stay inside a known physical range. Container conversion is useful for things
like species fractions, valve schedules, and groups of related values.
"""

from fullflow import *


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

# This pressure State is allowed to change, but it cannot go below zero.
pressure = State(300000.0, bounds=(0.0, None), keep_feasible=True)

# Bounds can also be added after the State is created.
mixture_ratio = State(2.3)
mixture_ratio.set_bounds((0.5, 8.0))

print("Pressure value:", pressure.value)
print("Pressure bounds:", pressure.bounds)
print("Mixture ratio value:", mixture_ratio.value)
print("Mixture ratio bounds:", mixture_ratio.bounds)


# ---------------------------------------------------------------------------
# Container conversion
# ---------------------------------------------------------------------------

# State.from_iterable() turns the values inside a container into States.
# This is helpful when the solver may need to change individual entries later.
composition = State.from_iterable({
    "O2": 0.21,
    "N2": 0.79,
})

# Each dictionary value is now a State.
print("\nO2 fraction:", composition["O2"].value)
print("N2 fraction:", composition["N2"].value)

# The individual entries can be changed normally.
composition["O2"].value = 0.20
composition["N2"].value = 0.80

print("Updated O2 fraction:", composition["O2"].value)
print("Updated N2 fraction:", composition["N2"].value)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

# A Network is the container for components, balances, and tracked outputs.
ExampleNetwork = Network("State Tracking Example")

# Tracked values are included in verbose summaries and HDF5 exports.
ExampleNetwork.track("Pressure [Pa]", pressure)
ExampleNetwork.track("Mixture Ratio", mixture_ratio)
ExampleNetwork.track("O2 Fraction", composition["O2"])
ExampleNetwork.track("N2 Fraction", composition["N2"])

# This example has no balances to solve. It only shows that the Network can
# collect the values you care about.
print("\nTracked outputs:")
for item in ExampleNetwork.tracked_state_list:
    print(item["name"])
