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
    The nonlinear solver still iterates on convenient thermodynamic variables:

        pressure
        pressure and enthalpy
        pressure and temperature

    but the integrated states are conservative extensive quantities:

        mass = density * volume
        total_internal_energy = mass * internal_energy

    This keeps the ROCETS/GFSSP-style pressure/enthalpy iteration behavior while
    allowing the volume itself to change during a transient. A scheduled or
    balanced volume can move independently of the pressure iteration variable,
    and mass is still conserved by the transient residual:

        d(mass)/dt = mass_flow_in - mass_flow_out

    If the energy balance is enabled, the component integrates total internal
    energy:

        d(mass * internal_energy)/dt =
            mass_flow_in * total_enthalpy_in
          - mass_flow_out * total_enthalpy_out
          + heat_rate
          + work_rate
          + boundary_work_rate

    where ``work_rate`` is an optional advanced user-supplied work input. Normal
    moving-boundary compression/expansion work is handled internally with:

        boundary_work=True

    using the backward-Euler volume change:

        boundary_work_rate = -work_pressure * (volume - volume.previous) / dt

    Positive heat or work adds energy to the control volume.

    Variable volume
    ---------------
    Set ``solve_volume=True`` when this node's geometric volume should be an
    additional transient unknown. The Volume does not add a separate volume
    residual; a normal Balance should close the geometry, for example:

        liquid_volume + ullage_volume - tank_volume = 0

    This keeps variable-volume tanks and moving interfaces user-facing as simple
    balances while the Volume hides the conservative mass/energy bookkeeping.
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
        work_rate: State | float | None = None,
        work_pressure: State | float | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        mass_flow_in: State | float | None = None,
        mass_flow_out: State | float | None = None,
        energy_variable: str = "enthalpy",
        mass: State | None = None,
        total_internal_energy: State | None = None,
        solve_volume: bool = False,
        boundary_work: bool = False,
        volume_derivative: State | None = None,
        boundary_work_rate: State | None = None,
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

        self._solve_volume = bool(solve_volume)
        self._boundary_work = bool(boundary_work)

        if self._solve_volume and volume is None:
            raise ValueError(f"{name}: solve_volume=True requires volume.")

        if self._boundary_work and volume is None:
            raise ValueError(f"{name}: boundary_work=True requires volume.")

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

        self._has_energy_output = (
            volume is not None
            and density is not None
            and internal_energy is not None
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
        self.solve_volume.value = self._solve_volume
        self.boundary_work.value = self._boundary_work

    def evaluate_states(self):
        if not self._has_mass_output:
            return

        self.mass.value = self.density.value * self.volume.value

        if self._has_energy_output:
            self.total_internal_energy.value = self.mass.value * self.internal_energy.value

        if self._boundary_work:
            self.volume_derivative.value = self._volume_derivative()
            self.boundary_work_rate.value = self._boundary_work_rate()

    def _outlet_enthalpy(self) -> float:
        if self.total_enthalpy_out.is_assigned:
            return self.total_enthalpy_out.value

        if self.enthalpy.is_assigned:
            return self.enthalpy.value

        raise ValueError(
            f"{self.name}: energy balance requires total_enthalpy_out or enthalpy to be assigned."
        )

    def _heat_rate(self) -> float:
        if self.heat_rate.is_assigned:
            return self.heat_rate.value
        return 0.0

    def _work_rate(self) -> float:
        if self.work_rate.is_assigned:
            return self.work_rate.value
        return 0.0

    def _work_pressure(self) -> float:
        if self.work_pressure.is_assigned:
            return self.work_pressure.value
        return self.pressure.value

    def _volume_derivative(self) -> float:
        if not self._boundary_work:
            return 0.0

        dt = float(getattr(self, "_transient_dt", 0.0))
        if dt <= 0.0:
            return 0.0

        try:
            return (self.volume.value - self.volume.previous) / dt
        except Exception:
            return 0.0

    def _boundary_work_rate(self) -> float:
        if not self._boundary_work:
            return 0.0

        return -self._work_pressure() * self._volume_derivative()

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

        h_out = self._outlet_enthalpy()
        energy_balance = (
            self.mass_flow_in.value * self.total_enthalpy_in.value
            - self.mass_flow_out.value * h_out
            + self._heat_rate()
            + self._work_rate()
        )

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
    def transient_algebraic_variables(self) -> list[State]:
        if self._solve_volume:
            return [self.volume]
        return []

    @property
    def transient_history_states(self) -> list[State]:
        if self._solve_volume or self._boundary_work:
            return [self.volume]
        return []

    @property
    def transient_states(self) -> list[State]:
        self._check_transient_mass_balance()
        self._check_transient_energy_balance()

        if not self._has_transient_energy_balance:
            return [self.mass]

        return [
            self.mass,
            self.total_internal_energy,
        ]

    @property
    def transient_derivatives(self) -> list[State | float]:
        self._check_transient_mass_balance()
        self._check_transient_energy_balance()

        mdot_in = self.mass_flow_in.value
        mdot_out = self.mass_flow_out.value

        mass_derivative = mdot_in - mdot_out

        if not self._has_transient_energy_balance:
            return [mass_derivative]

        h_out = self._outlet_enthalpy()

        total_internal_energy_derivative = (
            mdot_in * self.total_enthalpy_in.value
            - mdot_out * h_out
            + self._heat_rate()
            + self._work_rate()
            + self._boundary_work_rate()
        )

        return [
            mass_derivative,
            total_internal_energy_derivative,
        ]

