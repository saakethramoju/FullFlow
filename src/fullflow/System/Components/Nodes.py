from __future__ import annotations

from typing import TYPE_CHECKING
from numbers import Real

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State



class Solid(Component):
    """Lumped solid thermal node."""
    def __init__(
        self,
        name: str,
        network: Network,
        temperature: State,
        mass: float | None = None,
        specific_heat: State | None = None,
        characteristic_length: State | float | None = None,
        thermal_conductivity: State | float | None = None,
        convection_coefficient: State | float | None = None,
        biot_number: State | None = None,
        heat_rate: State | float = 0.0,
    ):
        self._has_biot_inputs = (
            characteristic_length is not None
            and thermal_conductivity is not None
            and convection_coefficient is not None
        )

        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.temperature]

    @property
    def residuals(self) -> list[float]:
        return [self.heat_rate.value]

    def evaluate_states(self):
        if not self._has_biot_inputs:
            return

        Lc = self.characteristic_length.value
        k = self.thermal_conductivity.value
        h = self.convection_coefficient.value

        self.biot_number.value = h * Lc / k







class Volume(Component):
    """Lumped steady-state fluid control volume."""

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: State | float,
        enthalpy: State | float | None = None,
        total_enthalpy_in: State | float | None = None,
        total_enthalpy_out: State | float | None = None,
        heat_rate: State | float | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        mass_flow_in: State | float | None = None,
        mass_flow_out: State | float | None = None,
    ):
        self._has_energy_balance = enthalpy is not None and total_enthalpy_in is not None
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        if self._has_energy_balance:
            return [self.pressure, self.enthalpy]

        return [self.pressure]

    @property
    def residuals(self) -> list[float]:
        mass_balance = self.mass_flow_in.value - self.mass_flow_out.value

        if not self._has_energy_balance:
            return [mass_balance]

        qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0
        h_out = self.total_enthalpy_out.value if self.total_enthalpy_out.is_assigned else self.enthalpy.value
        energy_balance = self.mass_flow_in.value * self.total_enthalpy_in.value - self.mass_flow_out.value * h_out + qdot

        return [mass_balance, energy_balance]