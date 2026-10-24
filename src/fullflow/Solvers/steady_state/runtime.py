"""Runtime network view used by the steady-state solver."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import copy
import math
from typing import Any

import numpy as np

from fullflow.System.State import (
    State,
    is_assignable_state_like,
    is_state_like,
    resolve_value,
)
from fullflow.Solvers.balance_filtering import filter_user_balances
from fullflow.Exceptions import SensorDataStop, SolverSetupError
from fullflow.Solvers.equations import (
    balance_object_equations,
    component_balances,
    component_dynamics,
    evaluate_components_for_equation_discovery,
)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class IterationItem:
    """One solver-controlled state plus diagnostic metadata."""

    state: Any
    label: str
    owner_kind: str


class RuntimeCache:
    """Cached solver metadata for one ``Network`` instance."""

    def __init__(self, network, ignore_balances=None) -> None:
        """Initialize the object and register any FullFlow state wiring."""
        self.network = network
        self.ignore_balances = ignore_balances
        self._active_sensor_balance_owner_ids: set[int] = set()
        self._direct_sensor_balance_owner_ids: set[int] = set()
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
        """Rebuild all cached lists, labels, and validation checks."""
        self.component_list = tuple(self.network.component_list)
        self.active_component_list = tuple(
            component
            for component in self.component_list
            if component.active_in_solver("steady")
        )
        self.balance_list, self.ignored_balance_list = filter_user_balances(
            self.network.balance_list,
            self.ignore_balances,
        )
        self.model_list = tuple(self.network.model_list)

        # Evaluate once before collecting equations created during evaluation.
        evaluate_components_for_equation_discovery(
            self.active_component_list,
            solve_mode="steady",
        )

        self.component_dynamic_items = tuple(self._dynamic_iteration_items())
        self.component_balance_items = tuple(self._component_balance_iteration_items())
        self.balance_items = tuple(self._balance_iteration_items())

        self.component_items = self.component_dynamic_items + self.component_balance_items
        self._validate_no_component_balance_overlap()

        self.iteration_items = self.component_items + self.balance_items
        self.iteration_variables = tuple(item.state for item in self.iteration_items)
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.iteration_labels = tuple(item.label for item in self.iteration_items)

        # Cache evaluation methods used by every residual call.
        self.pre_evaluation_callables = tuple(
            component.pre_evaluation for component in self.active_component_list
        )
        self.evaluate_state_callables = tuple(
            component.evaluate_states for component in self.active_component_list
        )

        # Track derived states for fixed-point convergence checks.
        self.state_refs = self._collect_state_refs()
        self.version = self.network.version

    @staticmethod
    def _is_sensor_anchor_owner(component: Any) -> bool:
        """Return True for Sensor-like components that can act as balances."""
        try:
            return bool(component.is_anchor)
        except Exception:
            return False

    @staticmethod
    def _sensor_extend_enabled(component: Any) -> bool:
        try:
            return bool(component._extend_enabled)
        except Exception:
            try:
                value = component.extend
                if is_state_like(value):
                    value = value.value
                return bool(value)
            except Exception:
                return True

    @staticmethod
    def _set_sensor_state(component: Any, name: str, value: Any) -> None:
        state = getattr(component, name, None)
        if is_state_like(state):
            state.value = value

    @staticmethod
    def _finite_number(value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return math.nan
        return number if math.isfinite(number) else math.nan

    def configure_sensor_balances(self, time_value: float, *, stop_on_missing: bool = True) -> None:
        """Freeze the active Sensor-balance set for one steady solve."""
        active_ids: set[int] = set()
        direct_ids: set[int] = set()

        for component in self.component_list:
            if not self._is_sensor_anchor_owner(component):
                continue

            component_id = id(component)
            target = self._finite_number(component.target_value(time_value))
            has_target = math.isfinite(target)

            variable = getattr(component, "variable", None)
            reading = getattr(component, "reading", None)
            variable_value = self._finite_number(getattr(variable, "numeric_value", math.nan))

            self._set_sensor_state(component, "data_value", target if has_target else math.nan)
            self._set_sensor_state(component, "variable_value", variable_value)

            if not has_target:
                self._set_sensor_state(component, "active", False)
                self._set_sensor_state(component, "error", math.nan)
                if stop_on_missing and not self._sensor_extend_enabled(component):
                    raise SensorDataStop(
                        f"Sensor {component.name!r} has no finite FullPlot Trace value at "
                        f"time {float(time_value):.9g}."
                    )
                continue

            if variable is reading:
                variable.value = target
                direct_ids.add(component_id)
                self._set_sensor_state(component, "active", True)
                self._set_sensor_state(component, "variable_value", target)
                self._set_sensor_state(component, "error", 0.0)
                continue

            active_ids.add(component_id)
            self._set_sensor_state(component, "active", True)
            try:
                self._set_sensor_state(component, "error", self._finite_number(reading.value - target))
            except Exception:
                self._set_sensor_state(component, "error", math.nan)

        self._active_sensor_balance_owner_ids = active_ids
        self._direct_sensor_balance_owner_ids = direct_ids
        self._rebuild_iteration_metadata()

    def _component_balance_equations(self, component: Any):
        """Return algebraic balances active for the current solve."""
        if not self._is_sensor_anchor_owner(component):
            return component_balances(component)
        component_id = id(component)
        if component_id in self._direct_sensor_balance_owner_ids:
            return []
        if component_id in self._active_sensor_balance_owner_ids:
            return component_balances(component)
        return []

    def _rebuild_iteration_metadata(self) -> None:
        """Rebuild algebraic and iteration lists after sensor activation changes."""
        self.component_balance_items = tuple(self._component_balance_iteration_items())
        self.balance_items = tuple(self._balance_iteration_items())
        self.component_items = self.component_dynamic_items + self.component_balance_items
        self._validate_no_component_balance_overlap()
        self.iteration_items = self.component_items + self.balance_items
        self.iteration_variables = tuple(item.state for item in self.iteration_items)
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.iteration_labels = tuple(item.label for item in self.iteration_items)
        self.state_refs = self._collect_state_refs()

    def _dynamic_iteration_items(self) -> list[IterationItem]:
        """Collect steady-state unknowns from component dynamics."""
        items: list[IterationItem] = []

        for component in self.active_component_list:
            for equation in component_dynamics(component):
                items.append(
                    IterationItem(
                        state=equation.variable,
                        label=self.state_label(component, equation.variable),
                        owner_kind="dynamic",
                    )
                )

        return items

    def _component_balance_iteration_items(self) -> list[IterationItem]:
        """Collect steady-state unknowns from component algebraic balances."""
        items: list[IterationItem] = []

        for component in self.active_component_list:
            for equation in self._component_balance_equations(component):
                items.append(
                    IterationItem(
                        state=equation.variable,
                        label=self.state_label(component, equation.variable),
                        owner_kind="component balance",
                    )
                )

        return items

    def _balance_iteration_items(self) -> list[IterationItem]:
        """Collect steady-state unknowns from user Balance objects."""
        items: list[IterationItem] = []

        for balance in self.balance_list:
            for equation in balance_object_equations(balance):
                items.append(
                    IterationItem(
                        state=equation.variable,
                        label=self.state_label(balance, equation.variable),
                        owner_kind="balance",
                    )
                )

        return items

    def _validate_no_component_balance_overlap(self) -> None:
        """Reject States solved by both a component and a Balance."""
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
            "A State used as a Balance solve variable must not also appear as a component dynamic or component balance variable.",
            "",
            "Overlapping component variables:",
            *[f"  - {name}" for name in sorted(component_names)],
            "",
            "Conflicting balance variables:",
            *[f"  - {name}" for name in sorted(balance_names)],
        ]
        raise SolverSetupError("\n".join(lines))

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

    def collect_residual_labels(self) -> list[str]:
        """Build labels matching the current residual vector order."""
        labels: list[str] = []

        for component in self.active_component_list:
            labels.extend(
                f"{component.name}.dynamic[{i}]"
                for i, _ in enumerate(component_dynamics(component))
            )
            labels.extend(
                f"{component.name}.balance[{i}]"
                for i, _ in enumerate(self._component_balance_equations(component))
            )

        for balance in self.balance_list:
            labels.extend(
                f"{balance.name}.balance[{i}]"
                for i, _ in enumerate(balance_object_equations(balance))
            )

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
            raise SolverSetupError(
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
        """Collect non-iteration State objects reachable from the network."""
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

    def collect_state_values_compact(self, out: list[float | None] | None = None) -> list[float | None]:
        refs = self.state_refs
        if out is None or len(out) != len(refs):
            out = [None] * len(refs)
        for index, state in enumerate(refs):
            try:
                value = float(state.value) if state.is_assigned else None
                out[index] = value if value is not None and math.isfinite(value) else None
            except (TypeError, ValueError, OverflowError, AttributeError):
                out[index] = None
        return out

    @staticmethod
    def max_state_change_compact(old: list[float | None], new: list[float | None]) -> float:
        change = 0.0
        for before, after in zip(old, new):
            if after is None:
                if before is not None:
                    return math.inf
                continue
            if before is None:
                change = max(change, abs(after))
            else:
                change = max(change, abs(after - before) / max(abs(after), 1.0))
        return change

    def snapshot_mutable_states(self):
        snapshot = []
        seen = set()
        for state in (*self.iteration_variables, *self.state_refs):
            if id(state) in seen:
                continue
            seen.add(id(state))
            if isinstance(state, State):
                snapshot.append(("state", state, self._copy_value(state._value),
                                 self._copy_value(state._previous),
                                 self._copy_value(state._second_previous)))
            elif is_assignable_state_like(state):
                try:
                    snapshot.append(("proxy", state, self._copy_value(state.value)))
                except (TypeError, ValueError, AttributeError):
                    continue
        return snapshot

    @staticmethod
    def _copy_value(value):
        if is_state_like(value):
            return value
        if isinstance(value, dict):
            return {RuntimeCache._copy_value(k): RuntimeCache._copy_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [RuntimeCache._copy_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(RuntimeCache._copy_value(item) for item in value)
        if isinstance(value, set):
            return {RuntimeCache._copy_value(item) for item in value}
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    @staticmethod
    def restore_mutable_states(snapshot) -> None:
        for item in snapshot:
            if item[0] == "state":
                _, state, value, previous, second_previous = item
                state._value = RuntimeCache._copy_value(value)
                state._previous = RuntimeCache._copy_value(previous)
                state._second_previous = RuntimeCache._copy_value(second_previous)
            else:
                _, state, value = item
                state.value = RuntimeCache._copy_value(value)

    @staticmethod
    def _append_residuals(residual_source: Any, append) -> None:
        if callable(residual_source):
            residual_source = residual_source()
        if residual_source is None:
            return
        if isinstance(residual_source, (list, tuple)):
            for value in residual_source:
                if callable(value):
                    value = value()
                append(float(resolve_value(value)))
            return
        append(float(resolve_value(residual_source)))

    def collect_state_values(self) -> dict[int, float]:
        """Return numeric values for assigned non-iteration states."""
        values: dict[int, float] = {}

        for state in self.state_refs:
            try:
                if state.is_assigned:
                    values[id(state)] = float(state.value)
            except Exception:
                # Ignore unavailable states during convergence monitoring.
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
        """Convert callable, State, or numeric residual values into floats."""
        values: list[float] = []
        RuntimeCache._append_residuals(residual_source, values.append)
        return values

    def collect_residuals(self) -> np.ndarray:
        """Evaluate and flatten all component and balance residuals."""
        residuals: list[float] = []
        append = residuals.append

        for component in self.active_component_list:
            try:
                # Steady state drives dynamic derivatives to zero.
                for equation in component_dynamics(component):
                    self._append_residuals(equation.derivative, append)

                # Algebraic balances are non-storage equations.
                for equation in self._component_balance_equations(component):
                    self._append_residuals(equation.residual, append)
            except Exception as error:
                original = str(error).splitlines()[0]
                raise SolverSetupError(
                    f"Failed while evaluating equations for component `{component.name}` "
                    f"of type `{type(component).__name__}`.\n\n"
                    "A State used inside this component's dynamics or balances is probably unassigned.\n\n"
                    "Likely fixes:\n"
                    "  - Give the missing State an initial value\n"
                    "  - Connect it to another component that computes it\n"
                    "  - Make it a dynamics or balances solve variable\n"
                    "  - Put conservation derivatives in dynamics and algebraic targets in balances\n\n"
                    f"Original error: {original}"
                ) from None

        for balance in self.balance_list:
            for equation in balance_object_equations(balance):
                self._append_residuals(equation.residual, append)

        return np.array(residuals, dtype=float)
