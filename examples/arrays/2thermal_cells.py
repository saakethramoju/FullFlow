"""
Transient Thermal Cells and Temperature-Dependent Stiffness
=========================================================

This example models eight independently integrated temperatures along a
vehicle wall. StateArray groups the scalar thermal States so their equations
and values can be handled as a single field, while the solver still integrates
eight ordinary scalar unknowns.

A second Component reads the complete temperature field to calculate a
stiffness distribution. The calculated stiffness is one array-valued State and
introduces no additional solver unknowns.
"""

import numpy as np
from fullflow import *


filename = "2thermal_cells"
dt = 0.1
t_final = 5.0


class ThermalCells(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        heat_input,
        thermal_capacity,
        heat_loss_coefficient: float,
    ):
        self.setup()

        shape = self.thermal_capacity.value.shape
        self.temperature = StateArray.full(shape, 293.15)
        self.temperature_rate = State(np.zeros(shape))
        self.stations = State(np.linspace(0.0, 1.0, shape[0]))

    def evaluate_states(self):
        heat_loss = self.heat_loss_coefficient.value * (
            self.temperature.value - 293.15
        )
        self.temperature_rate.value = (
            self.heat_input.value - heat_loss
        ) / self.thermal_capacity.value

    @property
    def dynamics(self):
        return [(self.temperature, self.temperature_rate)]


class ThermalStiffness(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        temperature: StateArray,
        reference_stiffness: float,
        temperature_coefficient: float,
        stiffness=None,
    ):
        self.setup()

    def evaluate_states(self):
        delta_temperature = self.temperature.value - 293.15
        self.stiffness.value = self.reference_stiffness.value * (
            1.0 - self.temperature_coefficient.value * delta_temperature
        )


# ---------------------------------------------------------------------------
# Network and boundary conditions
# ---------------------------------------------------------------------------

Vehicle = Network("Thermal Cells")

positions = np.linspace(0.0, 1.0, 8)
heat_input = 5.0 + 5.0 * positions
thermal_capacity = 1.0 + positions


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

NoseThermal = ThermalCells(
    "Nose Thermal",
    Vehicle,
    heat_input=heat_input,
    thermal_capacity=thermal_capacity,
    heat_loss_coefficient=0.08,
)

NoseStructure = ThermalStiffness(
    "Nose Structure",
    Vehicle,
    temperature=NoseThermal.temperature,
    reference_stiffness=3.0e6,
    temperature_coefficient=0.001,
)


# ---------------------------------------------------------------------------
# Outputs and transient solve
# ---------------------------------------------------------------------------

Vehicle.track("Mean Temperature [K]", NoseThermal.temperature.mean())
Vehicle.track("Tip Temperature [K]", NoseThermal.temperature.at(0))

Transient(Vehicle).solve(
    dt=dt,
    t_final=t_final,
    filename=filename,
    history="full",
)

print("Final temperatures [K]:", NoseThermal.temperature.value)
print("Final stiffness distribution [N m^2]:", NoseStructure.stiffness.value)
