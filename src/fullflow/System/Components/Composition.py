from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fullflow.System import Component, State
from fullflow.System.State import is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Composition(Component):
    """Conserve stream-carried scalar quantities in steady and transient networks."""

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
        self._amounts = {name: State(0.0) for name in self.names.value}

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
        return {str(name): value for name, value in solve.items()}

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

    def _composition_flow_rates(self, names: tuple[str, ...] | list[str]) -> dict[str, float]:
        rates = {name: 0.0 for name in names}
        if not rates:
            return rates
        for mass_flow, composition in self.inlets.value:
            flow = float(self._value(mass_flow))
            for name in names:
                rates[name] += flow * self._composition_value(composition, name)
        for mass_flow, composition in self.outlets.value:
            flow = float(self._value(mass_flow))
            for name in names:
                rates[name] -= flow * self._composition_value(composition, name)
        return rates

    def _composition_flow_rate(self, name: str) -> float:
        return self._composition_flow_rates((name,))[name]

    def _transient_names(self) -> tuple[str, ...]:
        if not self._has_transient_storage:
            return ()
        missing = [name for name in self.names.value if name not in self.solve.value]
        if missing:
            raise ValueError(
                f"{self.name}: transient composition requires solve states for "
                f"every transient label. Missing: {missing}."
            )
        non_states = [name for name in self.names.value if not is_state_like(self.solve.value[name])]
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
    def balances(self):
        if self._has_transient_storage:
            return []
        entries = [(name, value) for name, value in self.solve.value.items() if is_state_like(value)]
        rates = self._composition_flow_rates([name for name, _ in entries])
        return [(value, rates[name]) for name, value in entries]

    @property
    def dynamics(self):
        names = self._transient_names()
        rates = self._composition_flow_rates(names)
        return [(self.solve.value[name], self._amounts[name], rates[name]) for name in names]
