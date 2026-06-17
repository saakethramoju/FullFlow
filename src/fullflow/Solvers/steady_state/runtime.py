"""Runtime network view used by the steady-state solver.

``Network`` is intentionally kept as a simple container. This module holds the
solver-specific interpretation of that container: which states are iteration
variables, which residuals exist, what labels should be shown in diagnostics,
and which component methods need to be called.

The cache is rebuilt whenever ``network.version`` changes. That gives fast
residual evaluations during a solve while still supporting model replacement,
component registration, and future transient solvers that may build their own
runtime views.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fullflow.System.Composition import Composition
from fullflow.System.State import (
    is_assignable_state_like,
    is_state_like,
    resolve_value,
)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class IterationItem:
    """One solver-controlled state plus diagnostic metadata."""

    state: Any
    label: str
    owner_kind: str


class RuntimeCache:
    """Cached solver metadata for one ``Network`` instance.

    The cache is deliberately solver-side rather than network-side. ``Network``
    only stores components, balances, models, and tracked states. ``RuntimeCache``
    decides how those objects should be interpreted for steady-state solving.
    """

    def __init__(self, network) -> None:
        self.network = network
        self.version = -1
        self.refresh()

    def is_current(self) -> bool:
        """Return ``True`` if the cache matches the current network version."""
        return self.version == self.network.version

    def ensure_current(self) -> "RuntimeCache":
        """Refresh in-place if the network structure changed, then return self."""
        if not self.is_current():
            self.refresh()
        return self

    def refresh(self) -> None:
        """Rebuild all cached lists, labels, and validation checks.

        This is called before each top-level operation and whenever the solver
        detects a network version change. Keeping the expensive collection work
        here makes individual residual calls simpler and faster.
        """
        self.component_list = tuple(self.network.component_list)
        self.balance_list = tuple(self.network.balance_list)
        self.model_list = tuple(self.network.model_list)

        self.component_items = tuple(
            self._iteration_items(self.component_list, owner_kind="component")
        )
        self.balance_items = tuple(
            self._iteration_items(self.balance_list, owner_kind="balance")
        )
        self._validate_no_component_balance_overlap()

        self.iteration_items = self.component_items + self.balance_items
        self.iteration_variables = tuple(item.state for item in self.iteration_items)
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.iteration_labels = tuple(item.label for item in self.iteration_items)

        # Bound method lookup is cheap, but doing it once makes residual calls
        # easier to read and avoids repeatedly walking the component list.
        self.pre_evaluation_callables = tuple(
            component.pre_evaluation for component in self.component_list
        )
        self.evaluate_state_callables = tuple(
            component.evaluate_states for component in self.component_list
        )

        # Track non-iteration states so StateEvaluator can detect convergence of
        # derived-state fixed-point passes.
        self.state_refs = self._collect_state_refs()
        self.version = self.network.version

    def _iteration_items(
        self,
        owners: Iterable[Any],
        *,
        owner_kind: str,
    ) -> list[IterationItem]:
        """Collect and validate iteration variables from components or balances."""
        items: list[IterationItem] = []

        for owner in owners:
            for state in owner.iteration_variables:
                if not is_assignable_state_like(state):
                    raise TypeError(
                        f"{owner.name}: iteration variable must be an "
                        "assignable, non-derived State-like object."
                    )
                items.append(
                    IterationItem(
                        state=state,
                        label=self.state_label(owner, state),
                        owner_kind=owner_kind,
                    )
                )

        return items

    def _validate_no_component_balance_overlap(self) -> None:
        """Reject States solved by both a component and a Balance.

        A component iteration variable is owned by the component equations. A
        ``Balance`` variable is owned by a user algebraic target. Allowing the
        same State in both places would create two independent solve intents for
        one unknown and is almost always a modeling mistake.
        """
        component_ids = {id(item.state) for item in self.component_items}
        balance_ids = {id(item.state) for item in self.balance_items}
        overlap_ids = component_ids & balance_ids

        if not overlap_ids:
            return

        component_names = {
            item.label for item in self.component_items if id(item.state) in overlap_ids
        }
        balance_names = {
            item.label for item in self.balance_items if id(item.state) in overlap_ids
        }

        lines = [
            "Iteration variable overlap detected.",
            "",
            "Balance iteration variables cannot be the same as component iteration variables.",
            "A State used as a Balance solve variable must not also appear in a Component iteration_variables list.",
            "",
            "Overlapping component variables:",
            *[f"  - {name}" for name in sorted(component_names)],
            "",
            "Conflicting balance variables:",
            *[f"  - {name}" for name in sorted(balance_names)],
        ]
        raise ValueError("\n".join(lines))

    @staticmethod
    def state_label(owner: Any, state: Any) -> str:
        """Best-effort ``owner:attribute`` label for diagnostics."""
        for attr_name, attr_value in owner.__dict__.items():
            if attr_value is state:
                return f"{owner.name}:{attr_name}"

            if isinstance(attr_value, Composition):
                for species, species_state in attr_value.fraction.items():
                    if species_state is state:
                        return f"{owner.name}:{attr_name}.{species}"

        return f"{owner.name}:<unknown>"

    def find_variable_labels(self, target: Any) -> list[str]:
        """Return every owner attribute that references ``target``."""
        labels: list[str] = []

        for owner in self.component_list + self.balance_list:
            for attr_name, attr_value in owner.__dict__.items():
                if attr_value is target:
                    labels.append(f"{owner.name}.{attr_name}")

                if isinstance(attr_value, Composition):
                    for species, state in attr_value.fraction.items():
                        if state is target:
                            labels.append(f"{owner.name}.{attr_name}.{species}")

        return labels or [str(target)]

    def collect_residual_labels(self) -> list[str]:
        """Build labels matching the current residual vector order."""
        labels: list[str] = []

        for owner in self.component_list + self.balance_list:
            residuals = owner.residuals
            if isinstance(residuals, (list, tuple)):
                labels.extend(
                    f"{owner.name}.residual[{i}]" for i in range(len(residuals))
                )
            else:
                labels.append(f"{owner.name}.residual")

        return labels

    @property
    def iteration_values(self) -> list[float]:
        """Current numeric values of all iteration variables in solver order."""
        return [state.numeric_value for state in self.iteration_variables]

    @property
    def lower_bounds(self) -> list[float]:
        """Lower bounds aligned with ``iteration_values``."""
        return [state.lower_bound for state in self.iteration_variables]

    @property
    def upper_bounds(self) -> list[float]:
        """Upper bounds aligned with ``iteration_values``."""
        return [state.upper_bound for state in self.iteration_variables]

    @property
    def keep_feasible(self) -> list[bool]:
        """SciPy keep-feasible flags aligned with ``iteration_values``."""
        return [state.keep_feasible for state in self.iteration_variables]

    def assign_iteration_values(self, values: Iterable[float]) -> None:
        """Write a solver vector back into the iteration States."""
        values = list(values)
        if len(values) != len(self.iteration_variables):
            raise ValueError(
                f"Length mismatch: got {len(values)} iteration values "
                f"but expected {len(self.iteration_variables)}."
            )

        for value, state in zip(values, self.iteration_variables):
            state.value = value

    def snapshot_iteration_variables(self) -> tuple[tuple[Any, Any], ...]:
        """Capture assigned iteration values so derived-state passes can restore them."""
        return tuple(
            (state, state.value)
            for state in self.iteration_variables
            if state.is_assigned
        )

    @staticmethod
    def restore_iteration_variables(snapshot: tuple[tuple[Any, Any], ...]) -> None:
        """Restore a snapshot created by :meth:`snapshot_iteration_variables`."""
        for state, value in snapshot:
            state.value = value

    def run_pre_evaluation(self) -> None:
        """Run all component ``pre_evaluation()`` hooks in network order."""
        for pre_evaluation in self.pre_evaluation_callables:
            pre_evaluation()

    def _collect_state_refs(self) -> tuple[Any, ...]:
        """Collect non-iteration State-like objects reachable from the network.

        These references are used only for fixed-point convergence checks. The
        traversal includes common containers so components can store States in
        lists, tuples, dictionaries, and ``Composition`` objects.
        """
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

            if isinstance(value, dict):
                for key, item in value.items():
                    collect(key)
                    collect(item)
                return

            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    collect(item)
                return

        for owner in self.component_list + self.balance_list:
            for value in owner.__dict__.values():
                collect(value)

        return tuple(refs)

    def collect_state_values(self) -> dict[int, float]:
        """Return numeric values for assigned non-iteration states."""
        values: dict[int, float] = {}

        for state in self.state_refs:
            try:
                if state.is_assigned:
                    values[id(state)] = float(state.value)
            except Exception:
                # Non-numeric or temporarily unavailable States cannot be used
                # for convergence monitoring, but they may still be valid model
                # outputs. Ignore them here rather than failing the solve.
                continue

        return values

    @staticmethod
    def max_state_change(old: dict[int, float], new: dict[int, float]) -> float:
        """Largest relative change between two state-value snapshots."""
        max_change = 0.0

        for key, new_value in new.items():
            old_value = old.get(key, _MISSING)
            if old_value is _MISSING:
                max_change = max(max_change, abs(new_value))
                continue

            scale = max(abs(new_value), 1.0)
            max_change = max(max_change, abs(new_value - old_value) / scale)

        return max_change

    @staticmethod
    def flatten_residuals(residual_source: Any) -> list[float]:
        """Convert a component/balance residual return value into floats."""
        if residual_source is None:
            return []
        if not isinstance(residual_source, (list, tuple)):
            residual_source = (residual_source,)
        return [float(resolve_value(value)) for value in residual_source]

    def collect_residuals(self) -> np.ndarray:
        """Evaluate and flatten all component and balance residuals.

        Component residual failures get a richer message because they are most
        often caused by an unassigned State referenced inside a residual
        expression. Balance residuals are intentionally allowed to raise their
        original errors directly because they are user-provided functions.
        """
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
