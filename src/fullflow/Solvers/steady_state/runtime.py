from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from fullflow.System.Composition import Composition
from fullflow.System.State import State


_MISSING = object()


def is_state_like(value: Any) -> bool:
    """Return True for objects the solver can read/write like a State."""
    if isinstance(value, State):
        return True

    # CallableLookupAttribute intentionally behaves like a State but lives in
    # the Lookups package. Avoid importing it here so this module stays free of
    # circular imports during package initialization. Avoid hasattr(), because
    # CallableLookup dynamically creates attributes for unknown names.
    if value.__class__.__name__ == "CallableLookupAttribute":
        return True

    return any(
        "is_assigned" in getattr(cls, "__dict__", {})
        and "value" in getattr(cls, "__dict__", {})
        for cls in type(value).__mro__
    )


def state_value(value: Any) -> Any:
    return value.value if is_state_like(value) else value


class RuntimeCache:
    """Fast per-solve view of a network.

    The old solver repeatedly scanned every component attribute for every
    residual evaluation and every state-settling pass.  This cache keeps the
    same semantics but captures stable references once per solve/model option:

    * iteration-variable State-like objects
    * component evaluate_states() callables
    * component/balance residual owners
    * non-iteration State-like objects monitored during state propagation

    The cache is deliberately rebuilt at the beginning of every solve/static
    evaluation so model replacement and newly registered components are handled
    exactly as before.
    """

    def __init__(self, network) -> None:
        self.network = network
        self.refresh()

    def refresh(self) -> None:
        self.iteration_variables = tuple(self.network.collect_all_iteration_variables())
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.component_list = tuple(self.network.component_list)
        self.balance_list = tuple(self.network.balance_list)
        self.evaluate_state_callables = tuple(
            component.evaluate_states for component in self.component_list
        )
        self.state_refs = self._collect_state_refs()

    # ------------------------------------------------------------------
    # Iteration variables
    # ------------------------------------------------------------------

    @property
    def iteration_values(self) -> list[float]:
        return [state.numeric_value for state in self.iteration_variables]

    @property
    def lower_bounds(self) -> list[float]:
        return [state.lower_bound for state in self.iteration_variables]

    @property
    def upper_bounds(self) -> list[float]:
        return [state.upper_bound for state in self.iteration_variables]

    @property
    def keep_feasible(self) -> list[bool]:
        return [state.keep_feasible for state in self.iteration_variables]

    def assign_iteration_values(self, values: Iterable[float]) -> None:
        values = list(values)
        if len(values) != len(self.iteration_variables):
            raise ValueError(
                f"Length mismatch: got {len(values)} iteration values "
                f"but expected {len(self.iteration_variables)}."
            )

        for value, state in zip(values, self.iteration_variables):
            state.value = value

    def snapshot_iteration_variables(self) -> tuple[tuple[Any, Any], ...]:
        snapshot = []
        for state in self.iteration_variables:
            if state.is_assigned:
                snapshot.append((state, state.value))
        return tuple(snapshot)

    @staticmethod
    def restore_iteration_variables(snapshot: tuple[tuple[Any, Any], ...]) -> None:
        for state, value in snapshot:
            state.value = value

    # ------------------------------------------------------------------
    # State settling
    # ------------------------------------------------------------------

    def _collect_state_refs(self) -> tuple[Any, ...]:
        refs: list[Any] = []
        seen: set[int] = set()

        def add_state(state: Any) -> None:
            state_id = id(state)
            if state_id in self.iteration_ids or state_id in seen:
                return
            seen.add(state_id)
            refs.append(state)

        def collect(value: Any) -> None:
            if is_state_like(value):
                add_state(value)
                return

            if isinstance(value, Composition):
                for state in value.fraction.values():
                    add_state(state)
                return

            # Only traverse explicit built-in containers. Do not use hasattr()
            # or arbitrary iteration here: CallableLookup implements dynamic
            # __getattr__/__iter__, so probing it can accidentally manufacture
            # proxies or try to iterate its wrapped output.
            if isinstance(value, dict):
                for key, item in value.items():
                    collect(key)
                    collect(item)
                return

            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    collect(item)
                return

        for component in self.component_list:
            for value in component.__dict__.values():
                collect(value)

        for balance in self.balance_list:
            for value in balance.__dict__.values():
                collect(value)

        return tuple(refs)

    def collect_state_values(self) -> dict[int, float]:
        values: dict[int, float] = {}

        for state in self.state_refs:
            try:
                if not state.is_assigned:
                    continue
                values[id(state)] = float(state.value)
            except Exception:
                # Match the old behavior: state-settling diagnostics ignore
                # non-numeric/object states and states unavailable this pass.
                continue

        return values

    @staticmethod
    def max_state_change(old: dict[int, float], new: dict[int, float]) -> float:
        max_change = 0.0

        for key, new_value in new.items():
            old_value = old.get(key, _MISSING)

            if old_value is _MISSING:
                max_change = max(max_change, abs(new_value))
                continue

            scale = max(abs(new_value), 1.0)
            max_change = max(max_change, abs(new_value - old_value) / scale)

        return max_change

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    @staticmethod
    def flatten_residuals(residual_source) -> list[float]:
        if residual_source is None:
            return []
        if isinstance(residual_source, (list, tuple)):
            values = residual_source
        else:
            values = (residual_source,)

        residuals: list[float] = []
        for value in values:
            value = state_value(value)
            residuals.append(float(value))
        return residuals

    def collect_residuals(self) -> np.ndarray:
        residuals: list[float] = []

        for component in self.component_list:
            try:
                residuals.extend(self.flatten_residuals(component.residuals))
            except Exception as error:
                original = str(error).splitlines()[0]
                raise RuntimeError(
                    f"Failed while evaluating residuals for component `{component.name}` "
                    f"of type `{type(component).__name__}`.\n\n"
                    "A State used inside this component's residual equations is probably unassigned.\n\n"
                    "Likely fixes:\n"
                    "  - Give the missing State an initial value\n"
                    "  - Connect it to another component that computes it\n"
                    "  - Make it an iteration variable\n"
                    "  - If this is a static/transient-only quantity, do not use it in steady-state residuals\n\n"
                    f"Original error: {original}"
                ) from None

        for balance in self.balance_list:
            residuals.extend(self.flatten_residuals(balance.residuals))

        return np.array(residuals, dtype=float)
