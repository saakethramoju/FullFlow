from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Solid(Component):
    """Lumped solid thermal mass.

    ``Solid`` represents one thermal node.  The component has one simple
    conservation law:

        temperature_dot = heat_rate / (mass * specific_heat)

    If ``mass`` or ``specific_heat`` is not supplied, ``heat_rate`` is treated
    as an already-normalized temperature derivative.  This keeps the component
    useful for quick examples where the user wants to prescribe ``dT/dt``
    directly.

    Solver behavior
    ---------------
    Steady state:
        solve ``temperature_dot = 0``

    Transient:
        integrate ``d(temperature)/dt = temperature_dot``
    """

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

        # evaluate_states() overwrites this every pass.  Initializing it here
        # keeps the component safe to inspect before the first evaluation.
        self.temperature_dot = 0.0

        self.setup()

    def evaluate_states(self):
        if self.mass.is_assigned and self.specific_heat.is_assigned:
            self.temperature_dot = self.heat_rate.value / (self.mass.value * self.specific_heat.value)
        else:
            self.temperature_dot = self.heat_rate.value

        if not self._has_biot_inputs:
            return

        Lc = self.characteristic_length.value
        k = self.thermal_conductivity.value
        h = self.convection_coefficient.value

        self.biot_number.value = h * Lc / k

    @property
    def dynamics(self) -> list[tuple[State, float]]:
        return [(self.temperature, self.temperature_dot)]












class Volume(Component):
    """Lumped fluid control volume.

    ``Volume`` is the only fluid node/storage component.  There is no separate
    ``Junction`` class.  The same ``Volume`` can be used in two simple ways:

    1. Storage volume
       Provide ``volume`` and ``density``.  The component stores mass and uses
       ``dynamics``:

           mass = density * volume
           mass_dot = mass_flow_in - mass_flow_out

       The mass equation solves pressure.  Variable-geometry problems can still
       use a changing ``volume`` State, but the volume itself should come from a
       geometry relation, a user ``Balance``, or another component.

    2. Algebraic node
       Omit ``volume`` or ``density``.  The component has no storage, so it uses
       ``balances`` instead.  This preserves the old steady-state node behavior:

           mass_flow_in - mass_flow_out = 0

    If energy inputs are provided, ``Volume`` stores total internal energy:

        total_internal_energy = mass * internal_energy
        total_internal_energy_dot = energy_in - energy_out + heat + work

    Boundary work is automatic for energy-storage volumes:

        boundary_work_rate = -work_pressure * dV/dt

    ``work_rate`` remains available for explicit shaft, electrical, stirrer, or
    other non-flow work.  ``work_pressure`` defaults to the volume pressure.
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

        # Basic availability flags.
        #
        # Storage volumes should always own their mass conservation equation.
        # Omitted inlet/outlet flows are interpreted as zero-flow boundaries,
        # but the State objects created by Component.setup() can still be
        # connected later by another component.  This is what lets a blowdown
        # tank be written naturally as:
        #
        #     COPV = Volume(..., volume=..., density=...)
        #     Valve(..., mass_flow=COPV.mass_flow_out)
        #
        # The constructor sees mass_flow_out=None, but the Valve will assign the
        # generated COPV.mass_flow_out State during evaluation.
        self._has_mass_storage = volume is not None and density is not None
        self._has_mass_balance = self._has_mass_storage or (mass_flow_in is not None and mass_flow_out is not None)

        if self._energy_variable == "temperature":
            self._has_energy_variable = temperature is not None
        else:
            self._has_energy_variable = enthalpy is not None

        # Energy storage is enabled by providing a storage volume, an internal
        # energy lookup, and the selected thermodynamic solve variable.  Energy
        # transport terms may be omitted initially and connected later through
        # States such as COPV.total_enthalpy_out.
        self._has_energy_storage = self._has_mass_storage and internal_energy is not None and self._has_energy_variable

        # Algebraic energy balance mode is still supported for non-storage
        # nodes, but storage nodes use dynamics instead of balances.
        self._has_energy_balance = self._has_energy_storage or (
            not self._has_mass_storage
            and self._has_energy_variable
            and (
                total_enthalpy_in is not None
                or total_enthalpy_out is not None
                or enthalpy is not None
                or heat_rate is not None
                or work_rate is not None
            )
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

        # evaluate_states() overwrites these every pass.  Initializing them here
        # keeps dynamics/balances simple and avoids hidden getattr fallbacks.
        self.mass_dot = 0.0
        self.total_internal_energy_dot = 0.0

        self.setup()

        # These are stored as ordinary State attributes by Component.setup().
        # Keeping them visible in prints/HDF5 output makes model intent clear.
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
        # Storage mode: compute the extensive quantities from the current
        # thermodynamic state.  These are the quantities the transient solver
        # integrates when dynamics are active.
        if self._has_mass_storage:
            self.mass.value = self.density.value * self.volume.value

            if self._has_energy_storage:
                self.total_internal_energy.value = self.mass.value * self.internal_energy.value

            self.volume_derivative.value = self._volume_derivative()
            self.boundary_work_rate.value = self._boundary_work_rate()

        # The same conservation-law rates are used in both modes:
        # - storage mode: dynamics integrate them / steady drives them to zero
        # - algebraic mode: balances drive them to zero directly
        if self._has_mass_balance:
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

        energy = self._heat_rate() + self._work_rate() + self.boundary_work_rate.value

        if mass_flow_in != 0.0:
            if not self.total_enthalpy_in.is_assigned:
                raise ValueError(f"{self.name}: nonzero inlet flow requires total_enthalpy_in.")
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
        # Algebraic-node mode.  This preserves the old steady-state behavior for
        # a Volume that has mass/energy flow balances but no storage inventory.
        #
        # Storage mode returns no balances because dynamics already represent
        # the conservation laws.  SteadyState will drive those derivatives to
        # zero automatically.
        if not self._has_mass_balance or self._has_mass_storage:
            return []

        return list(zip(self._solve_variables(), self._steady_balance_residuals()))

    @property
    def dynamics(self) -> list[tuple[State, State, float]]:
        # Storage mode.  Three-entry dynamics mean:
        #
        #     (solve_variable, stored_quantity, derivative)
        #
        # Fluid storage always solves pressure from the mass inventory.  If a
        # model needs volume to move, provide volume as a State and close it
        # with a geometry/mechanical Balance.
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
