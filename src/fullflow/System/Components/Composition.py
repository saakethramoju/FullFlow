from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fullflow.System import Component, State
from fullflow.System.State import is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Composition(Component):
    """
    Conserves arbitrary quantities carried by mass flow.

    Steady state
    ------------
    For each conserved label, one residual is created:

        sum(mdot_in * x_in) - sum(mdot_out * x_out) = 0

    Transient
    ---------
    If mass is provided, the nonlinear solver iterates on the composition
    values in solve={...}, but the integrated states are stored amounts:

        amount_i = mass * x_i

    The transient equation is:

        d(amount_i)/dt = sum(mdot_in * x_in) - sum(mdot_out * x_out)

    This pairs naturally with a transient Volume that exposes node mass.
    """

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
        if not self._has_transient_storage:
            return

        mass = float(self.mass.value)

        for name in self._transient_names():
            self._amounts[name].value = mass * self._composition_value(None, name)

    @property
    def iteration_variables(self) -> list[State]:
        return [
            value
            for value in self.solve.value.values()
            if is_state_like(value)
        ]

    @property
    def residuals(self) -> list[State | float]:
        return [
            self._composition_flow_rate(name)
            for name in self.names.value
        ]

    @property
    def transient_variables(self) -> list[State]:
        return [
            self.solve.value[name]
            for name in self._transient_names()
        ]

    @property
    def transient_states(self) -> list[State]:
        return [
            self._amounts[name]
            for name in self._transient_names()
        ]

    @property
    def transient_derivatives(self) -> list[State | float]:
        return [
            self._composition_flow_rate(name)
            for name in self._transient_names()
        ]