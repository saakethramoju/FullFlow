from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Solid(Component):
    """Lumped solid thermal mass.

    ``Solid`` is a dynamic component.  It represents the thermal energy stored
    in a solid node, wall segment, chamber liner, heat-exchanger wall, or other
    lumped material.

    The physical equation is ordinary energy conservation:

        temperature_dot = heat_rate / (mass * specific_heat)

    where positive ``heat_rate`` adds energy to the solid.

    Solver behavior
    ---------------
    SteadyState drives ``temperature_dot`` to zero.
    Transient integrates ``temperature`` using ``temperature_dot``.

    Therefore ``Solid`` does not need an algebraic ``balance``.  Its steady-state
    equation is simply its dynamic derivative set to zero.
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

        # evaluate_states() overwrites this on every solver pass.  Initializing
        # it here keeps the component easy to inspect before the first solve.
        self.temperature_dot = 0.0

        self.setup()

    def evaluate_states(self):
        # If mass and specific heat are supplied, heat_rate is a heat flow [W]
        # and the derivative is dT/dt [K/s].  If they are omitted, heat_rate is
        # treated as an already-normalized derivative/error.  That preserves the
        # old steady-state convenience behavior while keeping one clear dynamic
        # equation.
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
    def dynamics(self):
        return [(self.temperature, self.temperature_dot)]








class Volume(Component):
    """Lumped fluid storage volume.

    ``Volume`` is a dynamic storage component.  It owns conservation-law
    derivatives and does not expose algebraic balances.

    Mass conservation::

        mass_dot = mass_flow_in - mass_flow_out

    Optional energy conservation::

        total_internal_energy_dot =
            mass_flow_in  * total_enthalpy_in
          - mass_flow_out * outlet_enthalpy
          + heat_rate
          + work_rate
          + boundary_work_rate

    SteadyState drives these derivatives to zero.  Transient integrates the
    stored extensive quantities.  This matches the ROCETS-style interpretation:
    storage, inertia, and capacitance belong in ``dynamics``; plain algebraic
    closure equations belong in ``balances`` or user ``Balance(...)`` objects.

    The solver can use convenient thermodynamic variables as unknowns while
    conserving extensive quantities internally.  For example, the solver can
    vary pressure while integrating mass, or vary enthalpy/temperature while
    integrating total internal energy.

    Changing volume and boundary work
    ---------------------------------
    ``volume`` may be a normal State, derived State, or scheduled State.  When
    ``boundary_work=True``, the component computes

        boundary_work_rate = -work_pressure * volume_dot

    where ``volume_dot`` is estimated from accepted transient history during a
    transient solve.  During steady-state trim, ``volume_dot`` is zero unless a
    future dedicated moving-boundary component supplies otherwise.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: State | float,
        density: State,
        temperature: State | None = None,
        enthalpy: State | float | None = None,
        internal_energy: State | None = None,
        mass_flow_in: State | float | None = None,
        mass_flow_out: State | float | None = None,
        total_enthalpy_in: State | float | None = None,
        total_enthalpy_out: State | float | None = None,
        heat_rate: State | float | None = None,
        work_rate: State | float | None = None,
        work_pressure: State | float | None = None,
        energy_variable: str = "enthalpy",
        solve_volume: bool = False,
        boundary_work: bool = False,
        mass: State | None = None,
        total_internal_energy: State | None = None,
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

        self.mass_dot = 0.0
        self.total_internal_energy_dot = 0.0

        self._has_energy = total_enthalpy_in is not None and (
            total_enthalpy_out is not None or enthalpy is not None
        )

        if self._has_energy and internal_energy is None:
            raise ValueError(f"{name}: energy dynamics require internal_energy.")

        if self._has_energy and self._energy_variable == "temperature" and temperature is None:
            raise ValueError(
                f"{name}: temperature must be provided when energy_variable='temperature'."
            )

        self.setup()

        # Optional output/diagnostic States start from harmless numerical
        # guesses.  The solver overwrites them during evaluate_states().
        # Initializing them avoids order-dependent crashes during the first
        # fixed-point evaluation pass when downstream components have not yet
        # produced their outputs.
        if not self.mass_flow_out.is_assigned:
            self.mass_flow_out.value = 0.0
        if not self.mass.is_assigned:
            self.mass.value = 0.0
        if not self.total_internal_energy.is_assigned:
            self.total_internal_energy.value = 0.0
        if not self.volume_derivative.is_assigned:
            self.volume_derivative.value = 0.0
        if not self.boundary_work_rate.is_assigned:
            self.boundary_work_rate.value = 0.0

        self.energy_variable.value = self._energy_variable
        self.solve_volume.value = self._solve_volume
        self.boundary_work.value = self._boundary_work

    def evaluate_states(self):
        # Stored mass is the extensive quantity integrated by the transient
        # solver.  Pressure is the convenient thermodynamic variable used to
        # close the nonlinear solve.
        self.mass.value = self.density.value * self.volume.value

        if self._has_energy:
            self.total_internal_energy.value = self.mass.value * self.internal_energy.value

        self.mass_dot = self.mass_flow_in.value - self.mass_flow_out.value

        if self._boundary_work:
            self.volume_derivative.value = self._volume_derivative()
            self.boundary_work_rate.value = -self._work_pressure() * self.volume_derivative.value
        else:
            self.volume_derivative.value = 0.0
            self.boundary_work_rate.value = 0.0

        if self._has_energy:
            self.total_internal_energy_dot = self._energy_derivative()

    def _outlet_enthalpy(self) -> float:
        if self.total_enthalpy_out.is_assigned:
            return self.total_enthalpy_out.value
        if self.enthalpy.is_assigned:
            return self.enthalpy.value
        raise ValueError(
            f"{self.name}: energy dynamics require total_enthalpy_out or enthalpy."
        )

    @staticmethod
    def _optional_value(state) -> float:
        """Return an optional energy source term.

        Omitted optional terms are zero.  Connected terms are evaluated normally.
        If a connected term depends on another component that has not evaluated
        yet, the unassigned-State error is allowed to propagate so the solver
        evaluator can defer this component and retry it later in the pass.
        """
        if state is None or not state.is_assigned:
            return 0.0
        return state.value

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

    def _energy_derivative(self) -> float:
        h_out = self._outlet_enthalpy()

        return (
            self.mass_flow_in.value * self.total_enthalpy_in.value
            - self.mass_flow_out.value * h_out
            + self._optional_value(self.heat_rate)
            + self._optional_value(self.work_rate)
            + self.boundary_work_rate.value
        )

    def _energy_solve_variable(self):
        if self._energy_variable == "temperature":
            return self.temperature
        return self.enthalpy

    @property
    def dynamics(self):
        # Three-entry dynamics mean:
        #
        #     (solve_variable, integrated_state, derivative)
        #
        # SteadyState drives ``derivative`` to zero.  Transient forms the
        # backward-Euler residual for ``integrated_state``.
        equations = [(self.pressure, self.mass, self.mass_dot)]

        if self._has_energy:
            equations.append(
                (
                    self._energy_solve_variable(),
                    self.total_internal_energy,
                    self.total_internal_energy_dot,
                )
            )

        return equations
