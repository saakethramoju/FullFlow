"""
Basic Balance example.

A Balance is the simplest way to add a custom algebraic equation to a FullFlow
network. A Balance has two parts:

    variable:
        The State the solver is allowed to change.

    function:
        The residual the solver drives to zero.

This example solves a simple ideal-gas pressure. The target density is known,
and the solver changes pressure until:

    density - target_density = 0
"""

from fullflow import *


GasNetwork = Network("Basic Balance Example")

# Known values.
temperature = State(300.0)          # K
gas_constant = State(287.0)         # J/kg-K
target_density = State(1.5)         # kg/m3

# Initial guess for the unknown pressure.
pressure = State(101325.0, bounds=(0.0, None), keep_feasible=True)

# Ideal-gas density. This is a derived State, so it updates whenever the solver
# changes pressure.
density = pressure / (gas_constant * temperature)

# The solver changes pressure until density equals target_density.
DensityBalance = Balance(
    "Density Balance",
    GasNetwork,
    variable=pressure,
    function=density - target_density,
)

GasNetwork.track("Pressure [Pa]", pressure)
GasNetwork.track("Temperature [K]", temperature)
GasNetwork.track("Density [kg/m3]", density)
GasNetwork.track("Target Density [kg/m3]", target_density)

SteadyState(GasNetwork).solve(verbose=True)
