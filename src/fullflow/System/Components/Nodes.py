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
        specific_heat: State | float | None = None,
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
    def residuals(self) -> list[State | float]:
        return [self.heat_rate.value]

    def evaluate_states(self):
        if not self._has_biot_inputs:
            return

        Lc = self.characteristic_length.value
        k = self.thermal_conductivity.value
        h = self.convection_coefficient.value

        self.biot_number.value = h * Lc / k

    @property
    def transient_variables(self) -> list[State]:
        return [self.temperature]

    @property
    def transient_derivatives(self) -> list[State | float]:
        return [self.heat_rate.value / (self.mass.value * self.specific_heat.value)]







class Volume(Component):
    """Lumped steady-state and transient fluid control volume.

    Steady state
    ------------
    The component solves ordinary algebraic mass and energy balances.

    Transient
    ---------
    The nonlinear solver iterates on pressure and either enthalpy or temperature,
    but the actual integrated states are density and internal energy.

    This follows the ROCETS-style idea:

        iteration variables:
            pressure, enthalpy

        integrated states:
            density, internal_energy

    This avoids using density/internal energy as nonlinear iteration variables
    while still integrating conservative mass and energy equations.

    The component also exposes:

        mass = density * volume

    so other transient components, like Composition, can integrate quantities
    such as m*x without needing to duplicate the node mass calculation.

    Assumptions
    -----------
    The current transient form assumes fixed volume. If volume changes with time,
    a volume_derivative term should be added later.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: State | float | None = None,
        enthalpy: State | float | None = None,
        total_enthalpy_in: State | float | None = None,
        total_enthalpy_out: State | float | None = None,
        heat_rate: State | float | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        mass_flow_in: State | float | None = None,
        mass_flow_out: State | float | None = None,
        energy_variable: str = "enthalpy",
        mass: State | None = None,
    ):
        aliases = {
            "h": "enthalpy",
            "enthalpy": "enthalpy",
            "t": "temperature",
            "temp": "temperature",
            "temperature": "temperature",
        }

        self._energy_variable = str(energy_variable).strip().lower()

        if self._energy_variable not in aliases:
            raise ValueError(
                f"{name}: energy_variable must be one of {tuple(sorted(aliases))}."
            )

        self._energy_variable = aliases[self._energy_variable]

        if self._energy_variable == "temperature":
            self._has_energy_balance = total_enthalpy_in is not None and (total_enthalpy_out is not None or enthalpy is not None)
        else:
            self._has_energy_balance = total_enthalpy_in is not None and enthalpy is not None

        if self._has_energy_balance and self._energy_variable == "temperature" and temperature is None:
            raise ValueError(
                f"{name}: temperature must be provided when energy_variable='temperature'."
            )

        self._has_mass_output = (
            volume is not None
            and density is not None
        )

        self._has_transient_mass_balance = (
            volume is not None
            and density is not None
            and mass_flow_in is not None
            and mass_flow_out is not None
        )

        self._has_transient_energy_balance = (
            self._has_transient_mass_balance
            and self._has_energy_balance
            and internal_energy is not None
        )

        self.setup()

        self.energy_variable.value = self._energy_variable

    def evaluate_states(self):
        if not self._has_mass_output:
            return

        self.mass.value = self.density.value * self.volume.value

    def _outlet_enthalpy(self) -> float:
        if self.total_enthalpy_out.is_assigned:
            return self.total_enthalpy_out.value

        if self.enthalpy.is_assigned:
            return self.enthalpy.value

        raise ValueError(
            f"{self.name}: energy balance requires total_enthalpy_out or enthalpy to be assigned."
        )

    def _check_transient_mass_balance(self) -> None:
        if not self._has_transient_mass_balance:
            raise ValueError(
                f"{self.name}: transient mass balance requires volume, density, "
                "mass_flow_in, and mass_flow_out."
            )

    def _check_transient_energy_balance(self) -> None:
        if self._has_energy_balance and not self._has_transient_energy_balance:
            raise ValueError(
                f"{self.name}: transient energy balance requires internal_energy "
                "in addition to the steady energy-balance inputs."
            )

    @property
    def iteration_variables(self) -> list[State]:
        if not self._has_energy_balance:
            return [self.pressure]

        if self._energy_variable == "temperature":
            return [self.pressure, self.temperature]

        return [self.pressure, self.enthalpy]

    @property
    def residuals(self) -> list[State | float]:
        mass_balance = self.mass_flow_in.value - self.mass_flow_out.value

        if not self._has_energy_balance:
            return [mass_balance]

        qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0
        h_out = self._outlet_enthalpy()
        energy_balance = self.mass_flow_in.value * self.total_enthalpy_in.value - self.mass_flow_out.value * h_out + qdot

        return [mass_balance, energy_balance]

    @property
    def transient_variables(self) -> list[State]:
        self._check_transient_mass_balance()
        self._check_transient_energy_balance()

        if not self._has_transient_energy_balance:
            return [self.pressure]

        if self._energy_variable == "temperature":
            return [self.pressure, self.temperature]

        return [self.pressure, self.enthalpy]

    @property
    def transient_states(self) -> list[State]:
        self._check_transient_mass_balance()
        self._check_transient_energy_balance()

        if not self._has_transient_energy_balance:
            return [self.density]

        return [
            self.density,
            self.internal_energy,
        ]

    @property
    def transient_derivatives(self) -> list[State | float]:
        self._check_transient_mass_balance()
        self._check_transient_energy_balance()

        V = self.volume.value
        rho = self.density.value

        mdot_in = self.mass_flow_in.value
        mdot_out = self.mass_flow_out.value

        density_derivative = (mdot_in - mdot_out) / V

        if not self._has_transient_energy_balance:
            return [density_derivative]

        u = self.internal_energy.value

        qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0
        h_out = self._outlet_enthalpy()

        energy_rate = (
            mdot_in * self.total_enthalpy_in.value
            - mdot_out * h_out
            + qdot
        )

        internal_energy_derivative = (energy_rate / V - u * density_derivative) / rho

        return [
            density_derivative,
            internal_energy_derivative,
        ]