"""
Manual Map example.

Map turns tabulated data into FullFlow output States. A manual Map is useful
when the data is small enough to write directly in the script.

This example creates a one-dimensional valve map:

    valve area -> discharge coefficient

The solver changes valve area until the mapped discharge coefficient reaches a
target value.
"""

import numpy as np

from fullflow import *


MapNetwork = Network("Manual Map Example")

valve_area = State(0.35, bounds=(0.0, 1.0), keep_feasible=True)
target_cd = State(0.74)

ValveMap = Map(
    "Valve Map",
    MapNetwork,
    inputs={
        "area": valve_area,
    },
    axes={
        "area": np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
    },
    outputs={
        "discharge_coefficient": np.array([0.20, 0.55, 0.75, 0.83, 0.86]),
    },
)

CdBalance = Balance(
    "Cd Balance",
    MapNetwork,
    variable=valve_area,
    function=ValveMap.discharge_coefficient - target_cd,
)

MapNetwork.track("Valve Area", valve_area)
MapNetwork.track("Discharge Coefficient", ValveMap.discharge_coefficient)
MapNetwork.track("Target Cd", target_cd)

SteadyState(MapNetwork).solve(verbose=True)
