from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State




class Solid(Component):
    """Lumped solid thermal node.

    ``Solid`` can be used in two ways:

    1. Thermal storage node
       Provide ``mass`` and ``specific_heat``.  The component integrates
       temperature using:

           temperature_dot = heat_rate / (mass * specific_heat)

    2. Algebraic thermal node
       Omit ``mass`` and ``specific_heat``.  The component has no thermal
       capacitance, so it cannot store energy.  Instead, it solves the
       quasi-steady heat balance:

           heat_rate = 0

       This is useful for massless thermal junctions or steady conduction
       interface temperatures.

    Solver behavior
    ---------------
    Steady state:
        Storage node: drive ``temperature_dot = 0``.
        Algebraic node: solve ``heat_rate = 0``.

    Transient:
        Storage node: integrate ``dT/dt = temperature_dot``.
        Algebraic node: solve ``heat_rate = 0`` at each accepted timestep.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        temperature: State,
        mass: State | float | None = None,
        specific_heat: State | float | None = None,
        characteristic_length: State | float | None = None,
        thermal_conductivity: State | float | None = None,
        convection_coefficient: State | float | None = None,
        biot_number: State | None = None,
        heat_rate: State | float = 0.0,
    ):
        self._has_thermal_storage = mass is not None and specific_heat is not None

        if (mass is None) != (specific_heat is None):
            raise ValueError(
                f"{name}: mass and specific_heat must be provided together. "
                "Provide both for thermal storage, or omit both for an algebraic thermal node."
            )

        self._has_biot_inputs = (
            characteristic_length is not None
            and thermal_conductivity is not None
            and convection_coefficient is not None
        )

        self.temperature_dot = 0.0

        self.setup()

    def evaluate_states(self):
        if self._has_thermal_storage:
            self.temperature_dot = self.heat_rate.value / (self.mass.value * self.specific_heat.value)
        else:
            self.temperature_dot = 0.0

        if not self._has_biot_inputs:
            return

        Lc = self.characteristic_length.value
        k = self.thermal_conductivity.value
        h = self.convection_coefficient.value

        self.biot_number.value = h * Lc / k

    @property
    def balances(self) -> list[tuple[State, float]]:
        if self._has_thermal_storage:
            return []

        return [(self.temperature, self.heat_rate.value)]

    @property
    def dynamics(self) -> list[tuple[State, float]]:
        if not self._has_thermal_storage:
            return []

        return [(self.temperature, self.temperature_dot)]










class Volume(Component):
    """Lumped fluid control volume.

    ``Volume`` is the only fluid node/storage component. There is no separate
    ``Junction`` class. The same ``Volume`` can be used in two simple ways:

    1. Storage volume
       Provide ``volume`` and ``density``. The component stores mass and uses
       ``dynamics``:

           mass = density * volume
           mass_dot = mass_flow_in - mass_flow_out

       The mass equation solves pressure.

    2. Algebraic node
       Omit ``volume`` or ``density``. The component has no storage, so it uses
       ``balances`` instead:

           mass_flow_in - mass_flow_out = 0

       This supports late connections such as:

           ChamberEnd = Volume(..., mass_flow_in=Pipe.mass_flow)
           Nozzle(..., mass_flow=ChamberEnd.mass_flow_out)

       The user does not need to create an explicit intermediate State.
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
        volume_derivative: State | None = None,
        boundary_work_rate: State | None = None,
    ):
        self._energy_variable = self._normalize_energy_variable(name, energy_variable)

        self._has_mass_storage = volume is not None and density is not None

        if self._energy_variable == "temperature":
            self._has_energy_variable = temperature is not None
        else:
            self._has_energy_variable = enthalpy is not None

        self._has_energy_storage = (
            self._has_mass_storage
            and internal_energy is not None
            and self._has_energy_variable
        )

        # Storage volumes use dynamics.
        # Algebraic volumes use balances.
        #
        # Do not require mass_flow_in and mass_flow_out to both be supplied
        # during Volume construction. FullFlow intentionally supports connecting
        # generated node states later through downstream components:
        #
        #     ChamberEnd = Volume(..., mass_flow_in=Pipe.mass_flow)
        #     Nozzle(..., mass_flow=ChamberEnd.mass_flow_out)
        #
        # After setup(), ChamberEnd.mass_flow_out exists as a State and can be
        # assigned by the nozzle.
        self._has_mass_balance = True

        # For algebraic nodes, energy balance should also not depend on whether
        # total_enthalpy_out was supplied during Volume construction. It may be
        # assigned later by a downstream component.
        self._has_energy_balance = self._has_energy_storage or (
            not self._has_mass_storage
            and self._has_energy_variable
        )

        if self._has_energy_balance and self._energy_variable == "temperature" and temperature is None:
            raise ValueError(
                f"{name}: temperature must be provided when energy_variable='temperature'."
            )

        if self._has_mass_storage and self._has_energy_variable and internal_energy is None:
            raise ValueError(
                f"{name}: transient energy storage requires internal_energy. "
                "For a steady-only algebraic node, omit volume or density."
            )

        self.mass_dot = 0.0
        self.total_internal_energy_dot = 0.0

        self.setup()

        self.energy_variable.value = self._energy_variable

    @staticmethod
    def _normalize_energy_variable(name: str, energy_variable: str) -> str:
        aliases = {
            "h": "enthalpy",
            "enthalpy": "enthalpy",
            "t": "temperature",
            "temp": "temperature",
            "temperature": "temperature",
        }

        value = str(energy_variable).strip().lower()

        if value not in aliases:
            raise ValueError(
                f"{name}: energy_variable must be one of {tuple(sorted(aliases))}."
            )

        return aliases[value]

    def evaluate_states(self):
        if self._has_mass_storage:
            self.mass.value = self.density.value * self.volume.value

            if self._has_energy_storage:
                self.total_internal_energy.value = self.mass.value * self.internal_energy.value

            self.volume_derivative.value = self._volume_derivative()
            self.boundary_work_rate.value = self._boundary_work_rate()

        self.mass_dot = self._optional_value(self.mass_flow_in) - self._optional_value(self.mass_flow_out)

        if self._has_energy_balance:
            self.total_internal_energy_dot = self._energy_derivative()

    def _outlet_enthalpy(self) -> float:
        if self.total_enthalpy_out.is_assigned:
            return self.total_enthalpy_out.value

        if self.enthalpy.is_assigned:
            return self.enthalpy.value

        raise ValueError(
            f"{self.name}: energy balance requires total_enthalpy_out or enthalpy."
        )

    @staticmethod
    def _optional_value(state) -> float:
        if state is None or not state.is_assigned:
            return 0.0

        return state.value

    def _heat_rate(self) -> float:
        return self._optional_value(self.heat_rate)

    def _work_rate(self) -> float:
        return self._optional_value(self.work_rate)

    def _work_pressure(self) -> float:
        if self.work_pressure.is_assigned:
            return self.work_pressure.value

        return self.pressure.value

    def _volume_derivative(self) -> float:
        dt = float(self._transient_dt)

        if dt <= 0.0:
            return 0.0

        try:
            return (self.volume.value - self.volume.previous) / dt
        except Exception:
            return 0.0

    def _boundary_work_rate(self) -> float:
        return -self._work_pressure() * self._volume_derivative()

    def _energy_derivative(self) -> float:
        mass_flow_in = self._optional_value(self.mass_flow_in)
        mass_flow_out = self._optional_value(self.mass_flow_out)

        energy = (
            self._heat_rate()
            + self._work_rate()
            + self._optional_value(self.boundary_work_rate)
        )

        if mass_flow_in != 0.0:
            if not self.total_enthalpy_in.is_assigned:
                raise ValueError(
                    f"{self.name}: nonzero inlet flow requires total_enthalpy_in."
                )

            energy += mass_flow_in * self.total_enthalpy_in.value

        if mass_flow_out != 0.0:
            energy -= mass_flow_out * self._outlet_enthalpy()

        return energy

    def _energy_variable_state(self) -> State:
        if self._energy_variable == "temperature":
            return self.temperature

        return self.enthalpy

    def _solve_variables(self) -> list[State]:
        if not self._has_energy_balance:
            return [self.pressure]

        if self._energy_variable == "temperature":
            return [self.pressure, self.temperature]

        return [self.pressure, self.enthalpy]

    def _steady_balance_residuals(self) -> list[float]:
        residuals = [self.mass_dot]

        if self._has_energy_balance:
            residuals.append(self.total_internal_energy_dot)

        return residuals

    @property
    def balances(self) -> list[tuple[State, float]]:
        # Algebraic-node mode.
        #
        # Storage volumes return no algebraic balances because their conservation
        # laws are represented through dynamics. SteadyState will drive those
        # derivatives to zero automatically.
        if self._has_mass_storage:
            return []

        return list(zip(self._solve_variables(), self._steady_balance_residuals()))

    @property
    def dynamics(self) -> list[tuple[State, State, float]]:
        # Storage mode.
        #
        # Three-entry dynamics mean:
        #
        #     (solve_variable, stored_quantity, derivative)
        if not self._has_mass_storage:
            return []

        equations = [
            (self.pressure, self.mass, self.mass_dot),
        ]

        if self._has_energy_storage:
            equations.append(
                (
                    self._energy_variable_state(),
                    self.total_internal_energy,
                    self.total_internal_energy_dot,
                )
            )

        return equations