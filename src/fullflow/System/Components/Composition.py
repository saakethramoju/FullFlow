from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fullflow.System import Component, State
from fullflow.System.State import is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Composition(Component):
    """Conservation component for arbitrary stream-carried scalar quantities.

        ``Composition`` balances labels such as species mass fraction, oxidizer
        fraction, mixture fraction, tracer concentration, or any scalar carried by
        mass flow.  Each inlet and outlet is a pair of ``(mass_flow, composition)``.
        A ``composition`` dictionary maps names to values; ``None`` means the stream
        uses the unknown composition values supplied in ``solve``.

        Steady state
        ------------
        For each label, the residual is ``sum(mdot_in*x_in) - sum(mdot_out*x_out)``.
        The solver varies the corresponding entry in ``solve``.

        Transient
        ---------
        If ``mass`` is supplied, the component integrates stored amounts ``mass*x``
        while still iterating the convenient composition values.  This pairs with a
        transient ``Volume`` that owns the node mass."""

    def __init__(
        self,
        name: str,
        network: Network,
        inlets: list[tuple[State | float, dict[str, State | float] | None]],
        outlets: list[tuple[State | float, dict[str, State | float] | None]],
        solve: dict[str, State | float] | None = None,
        names: tuple[str, ...] | list[str] | str | None = None,
        mass: State | float | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self._has_transient_storage = mass is not None

        self.setup()

        self.inlets.value = list(self.inlets.value)
        self.outlets.value = list(self.outlets.value)

        if self.solve.is_assigned:
            self.solve.value = self._normalize_solve(self.solve.value)
        else:
            self.solve.value = {}

        if self.names.is_assigned:
            self.names.value = self._normalize_names(self.names.value)
        else:
            self.names.value = tuple(self.solve.value)

        if not self.names.value:
            raise ValueError(
                f"{self.name}: no composition labels were provided. "
                "Pass names=(...) or solve={...}."
            )

        self._amounts = {
            name: State(0.0)
            for name in self.names.value
        }

    @staticmethod
    def _normalize_names(names: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
        if names is None:
            return ()

        if isinstance(names, str):
            return (names,)

        return tuple(str(name) for name in names)

    @staticmethod
    def _normalize_solve(solve: dict[str, State | float] | None) -> dict[str, State | float]:
        if solve is None:
            return {}

        return {
            str(name): value
            for name, value in solve.items()
        }

    @staticmethod
    def _value(value: State | float) -> Any:
        return value.value if is_state_like(value) else value

    def _composition_value(self, composition: dict[str, State | float] | None, name: str) -> float:
        if composition is None:
            if name not in self.solve.value:
                raise ValueError(
                    f"{self.name}: a stream uses None for {name!r}, "
                    f"but solve does not contain {name!r}."
                )

            return float(self._value(self.solve.value[name]))

        if name not in composition:
            return 0.0

        return float(self._value(composition[name]))

    def _composition_flow_rate(self, name: str) -> float:
        composition_flow_rate = 0.0

        for mass_flow, composition in self.inlets.value:
            composition_flow_rate += float(self._value(mass_flow)) * self._composition_value(composition, name)

        for mass_flow, composition in self.outlets.value:
            composition_flow_rate -= float(self._value(mass_flow)) * self._composition_value(composition, name)

        return composition_flow_rate

    def _transient_names(self) -> tuple[str, ...]:
        if not self._has_transient_storage:
            return ()

        missing = [
            name
            for name in self.names.value
            if name not in self.solve.value
        ]

        if missing:
            raise ValueError(
                f"{self.name}: transient composition requires solve states for "
                f"every transient label. Missing: {missing}."
            )

        non_states = [
            name
            for name in self.names.value
            if not is_state_like(self.solve.value[name])
        ]

        if non_states:
            raise ValueError(
                f"{self.name}: transient composition requires solve values to be States. "
                f"These labels are not States: {non_states}."
            )

        return self.names.value

    def evaluate_states(self) -> None:
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        if not self._has_transient_storage:
            return

        mass = float(self.mass.value)

        for name in self._transient_names():
            self._amounts[name].value = mass * self._composition_value(None, name)

    @property
    def balances(self):
        # Without storage mass, composition is an algebraic mixer/splitter.
        # The solver varies the requested composition values until each
        # composition flow-rate balance is zero.
        """Return algebraic equations contributed by this component.
        
                Each tuple is ``(iteration_variable, residual)``.  Steady-state and
                transient solvers vary the iteration variable until the residual is zero.
                Components without algebraic closure equations return an empty list or do
                not define this property."""
        if self._has_transient_storage:
            return []

        return [
            (value, self._composition_flow_rate(name))
            for name, value in self.solve.value.items()
            if is_state_like(value)
        ]

    @property
    def dynamics(self):
        # With storage mass, composition is a real dynamic inventory.  The solver
        # can vary mass fraction, but the transient state is stored amount:
        #
        #     amount_i = mass * x_i
        #     d(amount_i)/dt = inflow_i - outflow_i
        #
        # SteadyState drives the same derivative to zero.
        """Return dynamic equations contributed by this component.
        
                A two-item tuple ``(state, derivative)`` means the solver integrates that
                state directly.  A three-item tuple ``(iteration_state, stored_state,
                derivative)`` means the nonlinear solver iterates a convenient state but
                conserves/integrates a different stored quantity.  Steady-state solves
                drive the derivative to zero."""
        return [
            (self.solve.value[name], self._amounts[name], self._composition_flow_rate(name))
            for name in self._transient_names()
        ]

