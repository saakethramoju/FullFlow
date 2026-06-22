"""Runtime network view used by the transient solver.

``Network`` stays a lightweight container.  This module is the transient
solver's interpretation of that container: which States are integrated in time,
which States are algebraic unknowns, which component residuals are active, and
which State bounds should be passed to SciPy.

The transient unknown vector is ordered as

    transient variables
  + algebraic component iteration variables
  + balance iteration variables

A component can therefore fall into one of three useful categories:

* dynamic component
    Provides ``transient_variables`` and ``transient_derivatives``.  The solver
    turns those derivative equations into implicit integration residuals.

* residual algebraic component
    Provides normal steady-state ``iteration_variables`` and ``residuals``.
    These equations are solved at each new timestep.

* explicit algebraic component
    Provides no iteration variables and no residuals.  It is still evaluated on
    every residual call, but it adds no unknowns and no equations.  This matches
    components such as a discharge-coefficient calculation that directly writes
    a mass-flow State from the current pressure-drop guess.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from fullflow.System.State import (
    is_assignable_state_like,
    is_state_like,
    resolve_value,
)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class TransientItem:
    """One integrated State owned by one dynamic component.

    The derivative is intentionally **not** stored here.  Instead, the cache
    stores the derivative list index.  During every residual call, after the
    network has been evaluated at SciPy's current new-time guess, the solver
    re-reads ``owner.transient_derivatives[derivative_index]``.  That makes this
    API safe for both derivative States and direct expressions such as

        return [self.net_torque.value / self.polar_moment_of_inertia.value]

    because the expression is recomputed from the current guessed state.
    """

    state: Any
    derivative_index: int
    label: str
    derivative_label: str
    owner: Any


@dataclass(frozen=True, slots=True)
class IterationItem:
    """One algebraic solver-controlled State plus diagnostic metadata."""

    state: Any
    label: str
    owner_kind: str
    owner: Any


class TransientRuntimeCache:
    """Cached solver metadata for one transient ``Network`` instance.

    The cache is rebuilt whenever ``network.version`` changes.  Runtime metadata
    lives solver-side so transient rules do not complicate the base ``Network``
    class.
    """

    def __init__(self, network) -> None:
        self.network = network
        self.version = -1
        self.refresh()

    def is_current(self) -> bool:
        """Return ``True`` if this cache matches the current network version."""
        return self.version == self.network.version

    def ensure_current(self) -> "TransientRuntimeCache":
        """Refresh in-place if network structure changed, then return self."""
        if not self.is_current():
            self.refresh()
        return self

    def refresh(self) -> None:
        """Rebuild transient items, algebraic items, callables, and bounds."""
        self.component_list = tuple(self.network.component_list)
        self.balance_list = tuple(self.network.balance_list)
        self.model_list = tuple(self.network.model_list)

        self.schedule_components = tuple(
            component
            for component in self.component_list
            if bool(getattr(component, "_is_fullflow_schedule", False))
        )
        self.normal_components = tuple(
            component
            for component in self.component_list
            if not bool(getattr(component, "_is_fullflow_schedule", False))
        )

        self.transient_items = tuple(self._collect_transient_items())
        self.transient_variables = tuple(item.state for item in self.transient_items)
        self.transient_ids = {id(state) for state in self.transient_variables}

        self.algebraic_component_items = tuple(
            self._collect_algebraic_component_items()
        )
        self.balance_items = tuple(
            self._iteration_items(self.balance_list, owner_kind="balance")
        )

        self._validate_no_solver_variable_overlap()

        self.iteration_items = (
            self.transient_iteration_items
            + self.algebraic_component_items
            + self.balance_items
        )
        self.iteration_variables = tuple(item.state for item in self.iteration_items)
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.iteration_labels = tuple(item.label for item in self.iteration_items)

        self._validate_schedule_targets()

        # Schedules run first so commanded values are available to normal
        # components during every fixed-point pass and residual evaluation.
        self.pre_evaluation_callables = tuple(
            component.pre_evaluation for component in self.component_list
        )
        self.evaluate_state_callables = tuple(
            component.evaluate_states for component in self.schedule_components
        ) + tuple(
            component.evaluate_states for component in self.normal_components
        )

        self.algebraic_components = tuple(
            component
            for component in self.component_list
            if not self._component_is_dynamic(component)
            and not bool(getattr(component, "_is_fullflow_schedule", False))
        )

        # These references are used only to decide whether repeated
        # evaluate_states() passes have settled.  They exclude solver unknowns
        # so the evaluator never treats SciPy's current guess as a derived state.
        self.state_refs = self._collect_state_refs()
        self.version = self.network.version

    @property
    def transient_iteration_items(self) -> tuple[IterationItem, ...]:
        """Transient variables represented as iteration items for SciPy."""
        return tuple(
            IterationItem(
                state=item.state,
                label=item.label,
                owner_kind="transient",
                owner=item.owner,
            )
            for item in self.transient_items
        )

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Normalize component property outputs to a list.

        Users should return lists, but accepting tuples makes the API forgiving.
        ``None`` means an empty list.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        raise TypeError(
            "transient_variables and transient_derivatives must return a list or tuple."
        )

    def _component_transient_variables(self, component: Any) -> list[Any]:
        return self._as_list(component.transient_variables)

    def _component_transient_derivatives(self, component: Any) -> list[Any]:
        return self._as_list(component.transient_derivatives)

    def _component_is_dynamic(self, component: Any) -> bool:
        return len(self._component_transient_variables(component)) > 0

    @staticmethod
    def _is_valid_derivative(value: Any) -> bool:
        """Return True for supported derivative objects.

        A derivative may be a State-like object or a plain real number.  Booleans
        are rejected even though ``bool`` subclasses ``int`` because they are
        almost certainly a modeling error for dx/dt.
        """
        if is_state_like(value):
            return True
        if isinstance(value, bool):
            return False
        return isinstance(value, Real)

    def _validate_derivative(self, component: Any, derivative: Any, index: int) -> None:
        if self._is_valid_derivative(derivative):
            return
        raise TypeError(
            f"{component.name}: transient_derivatives[{index}] must be a State-like "
            f"object or a real number. Got {type(derivative).__name__}."
        )

    def _collect_transient_items(self) -> list[TransientItem]:
        """Collect integrated State/derivative pairs from dynamic components."""
        items: list[TransientItem] = []

        for component in self.component_list:
            variables = self._component_transient_variables(component)
            derivatives = self._component_transient_derivatives(component)

            if not variables and derivatives:
                raise ValueError(
                    f"{component.name}: transient_derivatives were provided, but "
                    "transient_variables is empty."
                )

            if len(variables) != len(derivatives):
                raise ValueError(
                    f"{component.name}: transient_variables and transient_derivatives "
                    f"must have the same length. Got {len(variables)} variables "
                    f"and {len(derivatives)} derivatives."
                )

            for index, (state, derivative) in enumerate(zip(variables, derivatives)):
                if not is_assignable_state_like(state):
                    raise TypeError(
                        f"{component.name}: transient variable {index} must be an "
                        "assignable, non-derived State-like object."
                    )

                self._validate_derivative(component, derivative, index)
                items.append(
                    TransientItem(
                        state=state,
                        derivative_index=index,
                        label=self.state_label(component, state),
                        derivative_label=self._derivative_label(component, derivative, index),
                        owner=component,
                    )
                )

        return items

    def _collect_algebraic_component_items(self) -> list[IterationItem]:
        """Collect iteration variables only from non-dynamic components.

        Dynamic components keep their steady-state API for normal ``SteadyState``
        solves, but during transient stepping their equations come from
        ``transient_variables`` and ``transient_derivatives`` instead.
        """
        owners = [
            component
            for component in self.component_list
            if not self._component_is_dynamic(component)
            and not bool(getattr(component, "_is_fullflow_schedule", False))
        ]
        return self._iteration_items(owners, owner_kind="component")

    def _iteration_items(
        self,
        owners: Iterable[Any],
        *,
        owner_kind: str,
    ) -> list[IterationItem]:
        """Collect and validate algebraic iteration variables."""
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
                        owner=owner,
                    )
                )

        return items

    def _validate_no_solver_variable_overlap(self) -> None:
        """Reject duplicate unknown ownership before SciPy sees the vector."""
        groups = {
            "transient": self.transient_iteration_items,
            "component": self.algebraic_component_items,
            "balance": self.balance_items,
        }

        owners_by_state: dict[int, list[str]] = {}
        for group_name, items in groups.items():
            for item in items:
                owners_by_state.setdefault(id(item.state), []).append(
                    f"{group_name}: {item.label}"
                )

        conflicts = [labels for labels in owners_by_state.values() if len(labels) > 1]
        if not conflicts:
            return

        lines = [
            "Transient solver variable overlap detected.",
            "",
            "Each State can be solved by only one transient equation, component equation, or Balance.",
            "Conflicting variables:",
        ]
        for labels in conflicts:
            lines.extend(f"  - {label}" for label in labels)
            lines.append("")
        raise ValueError("\n".join(lines).rstrip())

    def _validate_schedule_targets(self) -> None:
        """Reject schedules that try to prescribe a solver unknown.

        A scheduled value is known from time, so it cannot also be solved as a
        transient variable, algebraic iteration variable, or balance variable.
        """
        if not self.schedule_components:
            return

        solver_ids = {id(state) for state in self.iteration_variables}
        conflicts: list[str] = []

        for schedule in self.schedule_components:
            target = getattr(schedule, "target", None)
            if is_state_like(target) and id(target) in solver_ids:
                conflicts.append(self.state_label(schedule, target))

        if conflicts:
            lines = [
                "Schedule target overlap detected.",
                "",
                "A scheduled State is prescribed by time and cannot also be a solver unknown.",
                "Conflicting scheduled targets:",
                *[f"  - {label}" for label in conflicts],
            ]
            raise ValueError("\n".join(lines))

    @staticmethod
    def _state_paths(value: Any, target: Any, prefix: str) -> list[str]:
        paths: list[str] = []
        seen: set[int] = set()

        def collect(item: Any, path: str) -> None:
            item_id = id(item)
            if item_id in seen:
                return
            seen.add(item_id)

            if item is target:
                paths.append(path)
                return

            if is_state_like(item):
                try:
                    collect(item.value, path)
                except Exception:
                    pass
                return

            if isinstance(item, dict):
                for key, child in item.items():
                    collect(child, f"{path}.{key}")
                return

            if isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    collect(child, f"{path}[{index}]")
                return

        collect(value, prefix)
        return paths

    @classmethod
    def state_label(cls, owner: Any, state: Any) -> str:
        """Best-effort ``owner:attribute`` label for diagnostics."""
        for attr_name, attr_value in owner.__dict__.items():
            paths = cls._state_paths(attr_value, state, f"{owner.name}:{attr_name}")
            if paths:
                return paths[0]
        return f"{owner.name}:<unknown>"

    def find_variable_labels(self, target: Any) -> list[str]:
        """Return every owner attribute that references ``target``."""
        labels: list[str] = []

        for owner in self.component_list + self.balance_list:
            for attr_name, attr_value in owner.__dict__.items():
                labels.extend(
                    self._state_paths(attr_value, target, f"{owner.name}.{attr_name}")
                )

        return labels or [str(target)]

    def _derivative_label(self, owner: Any, derivative: Any, index: int) -> str:
        """Best-effort label for a derivative object."""
        if is_state_like(derivative):
            return self.state_label(owner, derivative)
        return f"{owner.name}:transient_derivatives[{index}]"

    def _current_derivative(self, item: TransientItem) -> Any:
        """Read the derivative value/property for ``item`` at the current state."""
        derivatives = self._component_transient_derivatives(item.owner)

        if len(derivatives) <= item.derivative_index:
            raise ValueError(
                f"{item.owner.name}: transient_derivatives changed length during the solve. "
                f"Expected at least {item.derivative_index + 1} entries."
            )

        derivative = derivatives[item.derivative_index]
        self._validate_derivative(item.owner, derivative, item.derivative_index)
        return derivative

    def collect_residual_labels(self) -> list[str]:
        """Build labels matching the current transient residual vector order."""
        labels: list[str] = []

        labels.extend(f"{item.label}.integration" for item in self.transient_items)

        for owner in self.algebraic_components:
            residuals = owner.residuals
            if isinstance(residuals, (list, tuple)):
                labels.extend(
                    f"{owner.name}.residual[{i}]" for i in range(len(residuals))
                )
            else:
                labels.append(f"{owner.name}.residual")

        for balance in self.balance_list:
            residuals = balance.residuals
            if isinstance(residuals, (list, tuple)):
                labels.extend(
                    f"{balance.name}.residual[{i}]" for i in range(len(residuals))
                )
            else:
                labels.append(f"{balance.name}.residual")

        return labels

    @property
    def iteration_values(self) -> list[float]:
        """Current numeric values of all new-time unknowns in solver order."""
        return [state.numeric_value for state in self.iteration_variables]

    @property
    def lower_bounds(self) -> list[float]:
        """Lower State bounds aligned with ``iteration_values``."""
        return [state.lower_bound for state in self.iteration_variables]

    @property
    def upper_bounds(self) -> list[float]:
        """Upper State bounds aligned with ``iteration_values``."""
        return [state.upper_bound for state in self.iteration_variables]

    @property
    def keep_feasible(self) -> list[bool]:
        """SciPy keep-feasible flags aligned with ``iteration_values``."""
        return [state.keep_feasible for state in self.iteration_variables]

    def assign_iteration_values(self, values: Iterable[float]) -> None:
        """Write a SciPy vector back into transient/algebraic/balance States."""
        values = list(values)
        if len(values) != len(self.iteration_variables):
            raise ValueError(
                f"Length mismatch: got {len(values)} iteration values "
                f"but expected {len(self.iteration_variables)}."
            )

        for value, state in zip(values, self.iteration_variables):
            state.value = value

    def snapshot_iteration_variables(self) -> tuple[tuple[Any, Any], ...]:
        """Capture assigned solver values so failed evaluations can be restored."""
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

    def store_previous_values(self) -> None:
        """Store accepted values for all transient variables.

        This is called only after a timestep is accepted.  During a nonlinear
        solve, ``state.previous`` stays fixed while SciPy changes ``state.value``.
        """
        for item in self.transient_items:
            item.state.store_previous()

    def clear_previous_values(self) -> None:
        """Clear transient history for all transient variables."""
        for item in self.transient_items:
            item.state.clear_previous()

    def _derivative_value(self, item: TransientItem) -> float:
        """Return the current numeric derivative for ``item``.

        The derivative is re-read from the owning component every time this is
        called.  That is what makes direct expression properties behave like
        derivative States: both are evaluated at the current new-time guess.
        """
        return float(resolve_value(self._current_derivative(item)))

    @staticmethod
    def _transient_scale(item: TransientItem) -> float:
        """Scale used to make integration residuals dimensionless-ish.

        Algebraic residuals are left exactly as the component returns them, just
        like steady-state.  Only internally generated integration residuals are
        scaled because their raw units could be kg, J, K, rad/s, etc.
        """
        return max(abs(float(item.state.value)), abs(float(item.state.previous)), 1.0)

    def collect_residuals(self, dt: float) -> np.ndarray:
        """Collect transient, algebraic, and balance residuals.

        Dynamic residuals are implicit backward-Euler residuals evaluated at the
        current guessed new-time state:

            state.value - state.previous - dt * derivative = 0

        The residual is divided by a simple state scale so ``rtol`` has a useful
        per-timestep meaning across different state units.
        """
        residuals: list[float] = []

        for item in self.transient_items:
            state = float(item.state.value)
            previous = float(item.state.previous)
            derivative = self._derivative_value(item)
            scale = self._transient_scale(item)
            residuals.append((state - previous - dt * derivative) / scale)

        for component in self.algebraic_components:
            residuals.extend(float(value) for value in component.residuals)

        for balance in self.balance_list:
            residuals.extend(float(value) for value in balance.residuals)

        return np.array(residuals, dtype=float)

    def validate_residual_count(self, residuals: np.ndarray) -> None:
        """Fail early if there are fewer equations than unknowns.

        Overconstrained systems are allowed because least-squares can minimize
        more residuals than unknowns.  Underdetermined systems are almost always
        missing component residuals, a balance, or a transient derivative.
        """
        if len(residuals) < len(self.iteration_variables):
            raise ValueError(
                "Transient solve requires at least as many residuals as solver variables. "
                f"Got {len(self.iteration_variables)} solver variables and "
                f"{len(residuals)} residuals.\n\n"
                "Common causes:\n"
                "  - A component lists iteration_variables but returns no residuals.\n"
                "  - A dynamic component is missing transient_derivatives.\n"
                "  - A Balance variable was added without enough equations."
            )

    def _collect_state_refs(self) -> tuple[Any, ...]:
        """Collect non-iteration State-like objects for fixed-point checks."""
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
                try:
                    collect(value.value)
                except Exception:
                    pass
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
        """Return numeric values for assigned non-iteration States."""
        values: dict[int, float] = {}

        for state in self.state_refs:
            try:
                if state.is_assigned:
                    values[id(state)] = float(state.value)
            except Exception:
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
