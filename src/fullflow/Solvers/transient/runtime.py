"""Cached network metadata and State operations for transient solves."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import copy
import math
from numbers import Real
from typing import Any

import numpy as np

from fullflow.System.Component import Component
from fullflow.System.State import (
    State,
    is_assignable_state_like,
    is_state_like,
    resolve_numeric,
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
class TransientItem:
    """One dynamic equation owned by one component."""

    variable: Any
    state: Any
    derivative_index: int
    variable_label: str
    state_label: str
    derivative_label: str
    owner: Any
    force_steady: bool = False

    @property
    def label(self) -> str:
        """Compact diagnostic label for this dynamic equation."""
        if self.variable is self.state:
            return self.variable_label
        return f"{self.variable_label} -> {self.state_label}"


@dataclass(frozen=True, slots=True)
class IterationItem:
    """One algebraic solver-controlled State plus diagnostic metadata."""

    state: Any
    label: str
    owner_kind: str
    owner: Any


class TransientRuntimeCache:
    """Cached solver metadata for one transient ``Network`` instance."""

    def __init__(
        self,
        network,
        ignore_balances=None,
        force_steady=None,
        force_steady_exceptions=None,
    ) -> None:

        """Initialize the helper and its solver state."""
        self.network = network
        self.ignore_balances = ignore_balances
        self.force_steady = force_steady
        self.force_steady_exceptions = force_steady_exceptions
        self._active_sensor_balance_owner_ids: set[int] = set()
        self._direct_sensor_balance_owner_ids: set[int] = set()
        self._sensor_balance_configured = False
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
        self.balance_list, self.ignored_balance_list = filter_user_balances(
            self.network.balance_list,
            self.ignore_balances,
        )
        self.model_list = tuple(self.network.model_list)


        self._transient_context_hooks = tuple(
            component.set_transient_context for component in self.component_list
        )
        self._apply_command_hooks = self._collect_component_hooks("apply_commands")
        self._commit_command_hooks = self._collect_component_hooks("commit_commands")
        self._clear_command_hooks = self._collect_component_hooks("clear_command_pending")


        self.set_transient_context(dt=0.0)
        self.apply_sequence_commands(float(self.network.time.value))


        evaluate_components_for_equation_discovery(
            self.component_list,
            solve_mode="transient",
        )

        self._cache_component_transient_lists()

        self.transient_items = tuple(self._collect_transient_items())
        self.transient_variables = tuple(item.variable for item in self.transient_items)
        self.transient_states = tuple(item.state for item in self.transient_items)
        self.transient_ids = {id(state) for state in self.transient_variables}
        self.transient_state_ids = {id(state) for state in self.transient_states}
        self.transient_history_states = tuple(
            self._collect_transient_history_states()
        )

        self.algebraic_component_items = tuple(
            self._collect_algebraic_component_items()
        )
        self.balance_items = tuple(self._collect_balance_items())

        self._validate_no_transient_state_overlap()

        raw_iteration_items = (
            self.transient_iteration_items
            + self.algebraic_component_items
            + self.balance_items
        )
        self.iteration_items = self._deduplicate_iteration_items(raw_iteration_items)
        self.iteration_variables = tuple(item.state for item in self.iteration_items)
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.iteration_labels = tuple(item.label for item in self.iteration_items)
        self.iteration_count = len(self.iteration_variables)

        self.pre_evaluation_callables = tuple(
            component.pre_evaluation for component in self.component_list
        )
        self.evaluate_state_callables = tuple(
            component.evaluate_states for component in self.component_list
        )


        self.algebraic_residual_owners = tuple(
            component for component in self.component_list
            if getattr(type(component), "balances", None) is not Component.balances
        )
        self.balance_residual_owners = self.balance_list


        self.state_refs = self._collect_state_refs()


        self.all_state_refs = self._collect_all_state_refs()
        self._history_state_refs = self._states_with_history()
        self.version = self.network.version

    @staticmethod
    def _is_sensor_anchor_owner(component: Any) -> bool:
        """Return True for Sensor-like components that can act as balances."""
        if getattr(type(component), "is_anchor", None) is None:
            return False
        try:
            return bool(component.is_anchor)
        except Exception:
            return False

    @staticmethod
    def _sensor_extend_enabled(component: Any) -> bool:
        """Return whether a sensor extends solver iteration."""
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
        """Set a sensor state while preserving its type."""
        state = getattr(component, name, None)
        if is_state_like(state):
            state.value = value

    @staticmethod
    def _finite_number(value: Any) -> float:
        """Return a finite scalar value or None."""
        try:
            number = float(value)
        except Exception:
            return math.nan
        return number if math.isfinite(number) else math.nan

    def configure_sensor_balances(self, time_value: float, *, stop_on_missing: bool = True) -> None:
        """Freeze the active Sensor-balance set for one nonlinear solve."""
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

        membership_changed = (
            not self._sensor_balance_configured
            or active_ids != self._active_sensor_balance_owner_ids
            or direct_ids != self._direct_sensor_balance_owner_ids
        )
        self._active_sensor_balance_owner_ids = active_ids
        self._direct_sensor_balance_owner_ids = direct_ids
        self._sensor_balance_configured = True
        if membership_changed:
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
        self.algebraic_component_items = tuple(
            self._collect_algebraic_component_items()
        )
        self.balance_items = tuple(self._collect_balance_items())

        self._validate_no_transient_state_overlap()

        raw_iteration_items = (
            self.transient_iteration_items
            + self.algebraic_component_items
            + self.balance_items
        )
        self.iteration_items = self._deduplicate_iteration_items(raw_iteration_items)
        self.iteration_variables = tuple(item.state for item in self.iteration_items)
        self.iteration_ids = {id(state) for state in self.iteration_variables}
        self.iteration_labels = tuple(item.label for item in self.iteration_items)
        self.iteration_count = len(self.iteration_variables)

        self.state_refs = self._collect_state_refs()
        self.all_state_refs = self._collect_all_state_refs()
        self._history_state_refs = self._states_with_history()

    def _cache_component_transient_lists(self) -> None:
        """Cache dynamic variable/state lists once per network version."""
        self._transient_variable_lists: dict[int, list[Any]] = {}
        self._transient_state_lists: dict[int, list[Any]] = {}
        self.dynamic_component_ids: set[int] = set()

        for component in self.component_list:
            dynamic_equations = component_dynamics(component)
            variables = [equation.variable for equation in dynamic_equations]
            states = [equation.state for equation in dynamic_equations]

            component_id = id(component)
            self._transient_variable_lists[component_id] = variables
            self._transient_state_lists[component_id] = states

            if variables:
                self.dynamic_component_ids.add(component_id)

    @property
    def transient_iteration_items(self) -> tuple[IterationItem, ...]:
        """Transient variables represented as iteration items for SciPy."""
        return tuple(
            IterationItem(
                state=item.variable,
                label=item.variable_label,
                owner_kind="transient",
                owner=item.owner,
            )
            for item in self.transient_items
        )

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Normalize component property outputs to a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        raise SolverSetupError("dynamics and balances must return a list or tuple.")

    @staticmethod
    def _component_label(component: Any) -> str:
        """Return a compact component diagnostic label."""
        return f"{component.name} ({type(component).__name__})"

    @staticmethod
    def _normalize_force_steady_string(value: str) -> str:
        """Normalize string controls and reject unsupported strings."""
        normalized = value.strip().lower()
        if normalized == "all":
            return normalized
        raise SolverSetupError(
            "force_steady must be None, 'all', or an iterable of Component objects. "
            f"Got string {value!r}."
        )

    @staticmethod
    def _is_component_iterable(value: Any) -> bool:
        """Return True for iterable selector containers, excluding strings."""
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes))

    def _normalize_component_iterable(self, value: Any, *, argument_name: str) -> tuple[Any, ...]:
        """Return a tuple of Component objects from a user selector."""
        if value is None:
            return ()

        if isinstance(value, (str, bytes)) or not self._is_component_iterable(value):
            raise SolverSetupError(
                f"{argument_name} must be an iterable of Component objects. "
                f"Got {type(value).__name__}."
            )

        components = tuple(value)
        for item in components:
            if not isinstance(item, Component):
                raise SolverSetupError(
                    f"{argument_name} must contain only Component objects. "
                    f"Got {type(item).__name__}: {item!r}."
                )

        return components

    def _validate_selected_components(self, components: tuple[Any, ...], *, argument_name: str) -> None:
        """Reject components that are not in the network or have no dynamics."""
        if not components:
            return

        network_component_ids = {id(component) for component in self.component_list}
        missing = [component for component in components if id(component) not in network_component_ids]
        if missing:
            lines = [
                f"{argument_name} contains components that are not registered with network {self.network.name!r}:",
            ]
            lines.extend(f"  - {self._component_label(component)}" for component in missing)
            raise SolverSetupError("\n".join(lines))

        nondynamic = [component for component in components if not self._component_is_dynamic(component)]
        if nondynamic:
            lines = [
                f"{argument_name} contains components with no dynamic equations:",
            ]
            lines.extend(f"  - {self._component_label(component)}" for component in nondynamic)
            raise SolverSetupError("\n".join(lines))

    def _force_steady_component_ids(self) -> set[int]:
        """Return component IDs whose dynamics should use derivative = 0."""
        force_steady = self.force_steady
        exceptions = self.force_steady_exceptions

        if isinstance(force_steady, str):
            force_steady = self._normalize_force_steady_string(force_steady)

        if force_steady is None:
            if exceptions is not None:
                raise SolverSetupError(
                    "force_steady_exceptions can only be used when force_steady='all'."
                )
            return set()

        if force_steady == "all":
            exception_components = self._normalize_component_iterable(
                exceptions,
                argument_name="force_steady_exceptions",
            )
            self._validate_selected_components(
                exception_components,
                argument_name="force_steady_exceptions",
            )
            exception_ids = {id(component) for component in exception_components}
            return {
                id(component)
                for component in self.component_list
                if self._component_is_dynamic(component) and id(component) not in exception_ids
            }

        if exceptions is not None:
            raise SolverSetupError(
                "force_steady_exceptions can only be used when force_steady='all'."
            )

        selected_components = self._normalize_component_iterable(
            force_steady,
            argument_name="force_steady",
        )
        self._validate_selected_components(
            selected_components,
            argument_name="force_steady",
        )
        return {id(component) for component in selected_components}

    def _component_transient_variables(self, component: Any) -> list[Any]:
        """Return a component's transient iteration variables."""
        return self._transient_variable_lists[id(component)]

    def _component_transient_states(self, component: Any) -> list[Any]:
        """Return a component's integrated transient states."""
        return self._transient_state_lists[id(component)]

    def _component_transient_derivatives(self, component: Any) -> list[Any]:


        """Return a component's transient derivatives."""
        return [equation.derivative for equation in component_dynamics(component)]

    def _component_is_dynamic(self, component: Any) -> bool:
        """Return whether a component contributes transient dynamics."""
        return id(component) in self.dynamic_component_ids

    @staticmethod
    def _is_valid_derivative(value: Any) -> bool:
        """Return True for supported derivative objects."""
        if is_state_like(value):
            return True
        if isinstance(value, bool):
            return False
        return isinstance(value, Real)

    def _validate_derivative(self, component: Any, derivative: Any, index: int) -> None:
        """Validate one component transient derivative."""
        if self._is_valid_derivative(derivative):
            return
        raise SolverSetupError(
            f"{component.name}: dynamics derivative[{index}] must be a State "
            f"or a numeric value. Got {type(derivative).__name__}."
        )

    def _collect_transient_items(self) -> list[TransientItem]:
        """Collect dynamic unknowns, integrated states, and derivatives."""
        items: list[TransientItem] = []
        force_steady_ids = self._force_steady_component_ids()

        for component in self.component_list:
            variables = self._component_transient_variables(component)
            states = self._component_transient_states(component)

            if not variables and states:
                raise SolverSetupError(
                    f"{component.name}: dynamic integrated states were provided, but "
                    "dynamic variables is empty."
                )

            if len(variables) != len(states):
                raise SolverSetupError(
                    f"{component.name}: dynamic variables and dynamic integrated states "
                    "must have the same length. Got "
                    f"{len(variables)} variables and {len(states)} states."
                )

            for index, (variable, state) in enumerate(zip(variables, states)):
                if not is_assignable_state_like(variable):
                    raise SolverSetupError(
                        f"{component.name}: dynamic variable {index} must be an "
                        "assignable, non-derived State."
                    )

                if not is_state_like(state):
                    raise SolverSetupError(
                        f"{component.name}: dynamic integrated state {index} must be a State."
                    )

                if not callable(getattr(state, "store_previous", None)):
                    raise SolverSetupError(
                        f"{component.name}: dynamic integrated state {index} must support "
                        "store_previous(). Use a State for dynamic integrated states."
                    )

                if not callable(getattr(state, "clear_previous", None)):
                    raise SolverSetupError(
                        f"{component.name}: dynamic integrated state {index} must support "
                        "clear_previous(). Use a State for dynamic integrated states."
                    )

                variable_label = self.state_label(component, variable)
                state_label = self.state_label(component, state)

                items.append(
                    TransientItem(
                        variable=variable,
                        state=state,
                        derivative_index=index,
                        variable_label=variable_label,
                        state_label=state_label,
                        derivative_label=f"{component.name}:dynamics[{index}]",
                        owner=component,
                        force_steady=id(component) in force_steady_ids,
                    )
                )

        return items

    def _collect_transient_history_states(self) -> list[Any]:
        """Collect integrated States whose previous values define BE residuals."""
        states: list[Any] = []
        seen: set[int] = set()

        def add(state: Any, owner: Any) -> None:
            """Add a unique transient-history state and its owner."""
            if not is_state_like(state):
                raise SolverSetupError(f"{owner.name}: dynamic integrated state must be State-like.")
            if not callable(getattr(state, "store_previous", None)):
                raise SolverSetupError(f"{owner.name}: dynamic integrated state must support store_previous().")
            state_id = id(state)
            if state_id in seen:
                return
            seen.add(state_id)
            states.append(state)

        for item in self.transient_items:
            add(item.state, item.owner)

        return states

    def _collect_algebraic_component_items(self) -> list[IterationItem]:
        """Collect variables from component algebraic balances."""
        items: list[IterationItem] = []

        for component in self.component_list:
            for equation in self._component_balance_equations(component):
                items.append(
                    IterationItem(
                        state=equation.variable,
                        label=self.state_label(component, equation.variable),
                        owner_kind="component balance",
                        owner=component,
                    )
                )

        return items

    def _collect_balance_items(self) -> list[IterationItem]:
        """Collect variables from user Balance objects."""
        items: list[IterationItem] = []

        for balance in self.balance_list:
            for equation in balance_object_equations(balance):
                items.append(
                    IterationItem(
                        state=equation.variable,
                        label=self.state_label(balance, equation.variable),
                        owner_kind="balance",
                        owner=balance,
                    )
                )

        return items

    @staticmethod
    def _deduplicate_iteration_items(items: tuple[IterationItem, ...]) -> tuple[IterationItem, ...]:
        """Return solver unknowns with shared State objects kept only once."""
        unique: list[IterationItem] = []
        seen: set[int] = set()

        for item in items:
            state_id = id(item.state)
            if state_id in seen:
                continue
            seen.add(state_id)
            unique.append(item)

        return tuple(unique)

    def _validate_no_transient_state_overlap(self) -> None:
        """Reject multiple integration equations for the same dynamic integrated state."""
        owners_by_state: dict[int, list[str]] = {}

        for item in self.transient_items:
            owners_by_state.setdefault(id(item.state), []).append(item.label)

        conflicts = [labels for labels in owners_by_state.values() if len(labels) > 1]
        if not conflicts:
            return

        lines = [
            "Transient state overlap detected.",
            "",
            "Each integrated transient State can appear in only one dynamic equation.",
            "Conflicting dynamic integrated states:",
        ]
        for labels in conflicts:
            lines.extend(f"  - {label}" for label in labels)
            lines.append("")
        raise SolverSetupError("\n".join(lines).rstrip())

    @staticmethod
    def _state_paths(value: Any, target: Any, prefix: str) -> list[str]:
        """Find object paths that reference a target state."""
        paths: list[str] = []
        seen: set[int] = set()

        def collect(item: Any, path: str) -> None:
            """Collect matching states or paths from nested values."""
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
        return f"{owner.name}:dynamics[{index}]"

    def _current_derivative(
        self,
        item: TransientItem,
        derivatives_by_owner: dict[int, list[Any]] | None = None,
    ) -> Any:
        """Read a derivative, optionally reusing this residual call's component list."""
        if derivatives_by_owner is None:
            derivatives = self._component_transient_derivatives(item.owner)
        else:
            owner_id = id(item.owner)
            if owner_id not in derivatives_by_owner:
                derivatives_by_owner[owner_id] = self._component_transient_derivatives(item.owner)
            derivatives = derivatives_by_owner[owner_id]

        if len(derivatives) <= item.derivative_index:
            raise SolverSetupError(
                f"{item.owner.name}: dynamics derivative changed length during the solve. "
                f"Expected at least {item.derivative_index + 1} entries."
            )

        derivative = derivatives[item.derivative_index]
        self._validate_derivative(item.owner, derivative, item.derivative_index)
        return derivative

    def collect_residual_labels(self) -> list[str]:
        """Build labels matching the current transient residual vector order."""
        labels: list[str] = []

        for item in self.transient_items:
            suffix = "steady" if item.force_steady else "integration"
            labels.append(f"{item.state_label}.{suffix}")

        for owner in self.algebraic_residual_owners:
            equations = self._component_balance_equations(owner)
            labels.extend(
                f"{owner.name}.balance[{i}]" for i in range(len(equations))
            )

        for balance in self.balance_residual_owners:
            equations = balance_object_equations(balance)
            labels.extend(
                f"{balance.name}.balance[{i}]" for i in range(len(equations))
            )

        return labels

    def dynamic_mode_rows(self) -> list[dict[str, Any]]:
        """Return dynamic-equation mode rows for diagnostics or metadata."""
        return [
            {
                "component": item.owner.name,
                "component_type": type(item.owner).__name__,
                "index": item.derivative_index,
                "variable": item.variable_label,
                "state": item.state_label,
                "mode": "steady" if item.force_steady else "dynamic",
            }
            for item in self.transient_items
        ]

    def dynamic_mode_summary(self) -> dict[str, Any]:
        """Return a compact HDF5-friendly summary of dynamic equation modes."""
        rows = self.dynamic_mode_rows()
        return {
            "force_steady_count": sum(row["mode"] == "steady" for row in rows),
            "dynamic_count": sum(row["mode"] == "dynamic" for row in rows),
            "dynamic_modes": "\n".join(
                f"{row['component']}.dynamic[{row['index']}]: {row['mode']}"
                for row in rows
            ),
        }

    @property
    def iteration_values(self) -> list[float]:
        """Current numeric values of all new-time unknowns in solver order."""
        return [state.numeric_value for state in self.iteration_variables]

    def iteration_value_array(self) -> np.ndarray:
        """Current numeric unknown vector for SciPy."""
        return np.fromiter(
            (state.numeric_value for state in self.iteration_variables),
            dtype=float,
            count=self.iteration_count,
        )

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

    def lower_bound_array(self) -> np.ndarray:
        """Lower State bounds as a NumPy array aligned with the unknown vector."""
        return np.fromiter(
            (state.lower_bound for state in self.iteration_variables),
            dtype=float,
            count=self.iteration_count,
        )

    def upper_bound_array(self) -> np.ndarray:
        """Upper State bounds as a NumPy array aligned with the unknown vector."""
        return np.fromiter(
            (state.upper_bound for state in self.iteration_variables),
            dtype=float,
            count=self.iteration_count,
        )

    def keep_feasible_array(self) -> np.ndarray:
        """SciPy keep-feasible flags aligned with the unknown vector."""
        return np.fromiter(
            (state.keep_feasible for state in self.iteration_variables),
            dtype=bool,
            count=self.iteration_count,
        )

    def assign_iteration_values(self, values: Iterable[float]) -> None:
        """Write a SciPy vector back into transient/algebraic/balance States."""
        try:
            value_count = len(values)  # type: ignore[arg-type]
        except TypeError:
            values = tuple(values)
            value_count = len(values)

        if value_count != self.iteration_count:
            raise SolverSetupError(
                f"Length mismatch: got {value_count} iteration values "
                f"but expected {self.iteration_count}."
            )

        for value, state in zip(values, self.iteration_variables):
            state.value = value

    @staticmethod
    def _snapshot_value(value: Any) -> Any:
        """Copy a stored State value while preserving State object references."""
        if is_state_like(value):
            return value

        if isinstance(value, dict):
            return {
                TransientRuntimeCache._snapshot_value(key): TransientRuntimeCache._snapshot_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [TransientRuntimeCache._snapshot_value(item) for item in value]

        if isinstance(value, tuple):
            return tuple(TransientRuntimeCache._snapshot_value(item) for item in value)

        if isinstance(value, set):
            return {TransientRuntimeCache._snapshot_value(item) for item in value}

        if isinstance(value, frozenset):
            return frozenset(
                TransientRuntimeCache._snapshot_value(item)
                for item in value
            )

        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def snapshot_mutable_states(self) -> tuple[tuple[Any, ...], ...]:
        """Capture all reachable mutable State values before a timestep attempt."""
        snapshot: list[tuple[Any, ...]] = []

        for state in self.all_state_refs:
            if isinstance(state, State):
                snapshot.append(
                    (
                        "state",
                        state,
                        self._snapshot_value(state._value),
                        self._snapshot_value(state._previous),
                        self._snapshot_value(state._second_previous),
                    )
                )
                continue

            if not is_assignable_state_like(state):
                continue

            try:
                if not state.is_assigned:
                    continue
                value = self._snapshot_value(state.value)
            except Exception:
                continue

            snapshot.append(("proxy", state, value))

        return tuple(snapshot)

    @staticmethod
    def restore_mutable_states(snapshot: tuple[tuple[Any, ...], ...]) -> None:
        """Restore a snapshot created by :meth:`snapshot_mutable_states`."""
        for item in snapshot:
            kind = item[0]

            if kind == "state":
                _, state, value, previous, second_previous = item
                state._value = TransientRuntimeCache._snapshot_value(value)
                state._previous = TransientRuntimeCache._snapshot_value(previous)
                state._second_previous = TransientRuntimeCache._snapshot_value(second_previous)
                continue

            _, state, value = item

            if not is_assignable_state_like(state):
                continue

            state.value = TransientRuntimeCache._snapshot_value(value)

    def snapshot_iteration_variables(self) -> tuple[tuple[Any, Any], ...]:
        """Capture assigned solver values so failed evaluations can be restored."""
        return tuple(
            (state, state.value)
            for state in self.iteration_variables
            if state.is_assigned
        )

    @staticmethod
    def restore_iteration_variables(snapshot: tuple[tuple[Any, Any], ...]) -> None:
        """Restore the trial vector, skipping assignments that cannot change a State."""
        for state, value in snapshot:
            if (
                type(state) is State
                and type(value) is float
                and state._expr is None
                and not (value < state._lower_bound or value > state._upper_bound)
            ):
                current = state._value
                if current is value:
                    continue
                if type(current) is float and current == value:
                    if value != 0.0 or math.copysign(1.0, current) == math.copysign(1.0, value):
                        continue
            state.value = value

    def _collect_component_hooks(self, method_name: str) -> tuple[tuple[Any, Any], ...]:
        """Cache any callable class hook without class/type-specific branching."""
        hooks = []
        for component in self.component_list:
            method = getattr(type(component), method_name, None)
            if callable(method):
                hooks.append((component, method))
        return tuple(hooks)

    def run_pre_evaluation(self) -> None:
        """Run all component ``pre_evaluation()`` hooks in network order."""
        for pre_evaluation in self.pre_evaluation_callables:
            pre_evaluation()

    def apply_sequence_commands(self, time_value: float | None = None) -> None:
        """Apply Sequence command traces before physical model evaluation."""
        if time_value is None:
            try:
                time_value = float(self.network.time.value)
            except Exception:
                time_value = 0.0

        current_time = float(time_value)
        for component, apply_commands in self._apply_command_hooks:
            apply_commands(component, current_time)

    def commit_sequence_commands(self) -> None:
        """Commit command samples after a timestep is accepted."""
        for component, commit_commands in self._commit_command_hooks:
            commit_commands(component)

    def clear_sequence_command_pending(self) -> None:
        """Clear unaccepted command samples after a failed timestep."""
        for component, clear_pending in self._clear_command_hooks:
            clear_pending(component)

    def set_transient_context(self, *, dt: float) -> None:
        """Pass timestep context to every component before evaluation."""
        for set_context in self._transient_context_hooks:
            set_context(dt=dt)

    def _states_with_history(self) -> tuple[Any, ...]:
        """States whose accepted previous values should be available next step."""
        states: list[Any] = []
        seen: set[int] = set()

        for state in self.transient_history_states + self.all_state_refs:
            if not is_state_like(state):
                continue
            if not callable(getattr(state, "store_previous", None)):
                continue
            state_id = id(state)
            if state_id in seen:
                continue
            seen.add(state_id)
            states.append(state)

        return tuple(states)

    def store_previous_values(self) -> None:
        """Store accepted values after initialization or an accepted timestep."""
        for state in self._history_state_refs:
            state.store_previous()

    def clear_previous_values(self) -> None:
        """Clear stored accepted-step history."""
        for state in self._history_state_refs:
            state.clear_previous()

    def _derivative_value(
        self,
        item: TransientItem,
        derivatives_by_owner: dict[int, list[Any]] | None = None,
    ) -> float:
        """Return a derivative evaluated at the current nonlinear trial state."""
        return resolve_numeric(self._current_derivative(item, derivatives_by_owner))

    @staticmethod
    def _append_residuals(residual_source: Any, append) -> None:
        """Append normalized residual values from a source."""
        if callable(residual_source):
            residual_source = residual_source()
        if residual_source is None:
            return
        if isinstance(residual_source, (list, tuple)):
            for value in residual_source:
                if callable(value):
                    value = value()
                append(resolve_numeric(value))
            return
        if callable(residual_source):
            residual_source = residual_source()
        append(resolve_numeric(residual_source))

    @staticmethod
    def flatten_residuals(residual_source: Any) -> list[float]:
        """Convert callable, State, or numeric residuals into floats."""
        values: list[float] = []
        TransientRuntimeCache._append_residuals(residual_source, values.append)
        return values

    @staticmethod
    def _transient_scale(
        item: TransientItem,
        *,
        dt: float,
        derivative: float,
    ) -> float:
        """Scale used for internally generated integration residuals."""
        return max(
            abs(float(item.state.value)),
            abs(float(item.state.previous)),
            abs(float(dt * derivative)),
            1.0,
        )

    def collect_residuals(self, dt: float) -> np.ndarray:
        """Collect transient, algebraic, and balance residuals."""
        residuals: list[float] = []
        append = residuals.append
        append_residuals = self._append_residuals
        derivatives_by_owner: dict[int, list[Any]] = {}

        for item in self.transient_items:
            derivative = self._derivative_value(item, derivatives_by_owner)
            if item.force_steady:
                append(derivative)
                continue

            state = float(item.state.value)
            previous = float(item.state.previous)
            flux = dt * derivative
            scale = max(abs(state), abs(previous), abs(flux), 1.0)
            append((state - previous - flux) / scale)

        for component in self.algebraic_residual_owners:
            for equation in self._component_balance_equations(component):
                append_residuals(equation.residual, append)

        for balance in self.balance_residual_owners:
            append_residuals(balance._residual, append)

        return np.asarray(residuals, dtype=float)

    def validate_residual_count(self, residuals: np.ndarray) -> None:
        """Fail early if there are fewer equations than unknowns."""
        if len(residuals) < len(self.iteration_variables):
            raise SolverSetupError(
                "Transient solve requires at least as many residuals as solver variables. "
                f"Got {len(self.iteration_variables)} solver variables and "
                f"{len(residuals)} residuals.\n\n"
                "Common causes:\n"
                "  - A component balance lists a variable but no residual.\n"
                "  - A dynamic component is missing a derivative in dynamics.\n"
                "  - A Balance variable was added without enough equations."
            )

    def _collect_all_state_refs(self) -> tuple[Any, ...]:
        """Collect every State-like object reachable from the network."""
        refs: list[Any] = []
        seen: set[int] = set()

        def add_state(state: Any) -> None:
            """Add a unique state reference to the collection."""
            state_id = id(state)
            if state_id in seen:
                return
            seen.add(state_id)
            refs.append(state)

        def collect(value: Any) -> None:
            """Collect matching states or paths from nested values."""
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

        collect(self.network.time)
        for owner in self.component_list + self.balance_list + self.model_list:
            for value in getattr(owner, "__dict__", {}).values():
                collect(value)

        return tuple(refs)

    def _collect_state_refs(self) -> tuple[Any, ...]:
        """Collect non-iteration State objects for fixed-point checks."""
        refs: list[Any] = []
        seen: set[int] = set()

        def add_state(state: Any) -> None:
            """Add a unique state reference to the collection."""
            state_id = id(state)
            if state_id in self.iteration_ids or state_id in seen:
                return
            seen.add(state_id)
            refs.append(state)

        def collect(value: Any) -> None:
            """Collect matching states or paths from nested values."""
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
    def _settling_value(state: Any) -> float | None:
        """Return a comparable value for derived-state settling."""
        try:
            if type(state) is State and state._expr is None:
                value = state._value
                return float(value) if value is not None else None
            return float(state.value) if state.is_assigned else None
        except Exception:
            return None

    def _collect_state_values_compact(self) -> list[float | None]:
        """Snapshot non-iteration States in cached reference order."""
        read_value = self._settling_value
        return [read_value(state) for state in self.state_refs]

    def _collect_state_values_compact_change(
        self,
        previous_values: list[float | None],
        output_values: list[float | None],
    ) -> float:
        """Update a reusable snapshot while calculating the fixed-point change."""
        max_change = 0.0
        read_value = self._settling_value
        for index, state in enumerate(self.state_refs):
            value = read_value(state)
            output_values[index] = value
            if value is None:
                continue

            previous = previous_values[index]
            if previous is None:
                change = abs(value)
            else:
                scale = max(abs(value), 1.0)
                change = abs(value - previous) / scale

            if change > max_change:
                max_change = change

        return max_change

    @staticmethod
    def _max_compact_state_change(
        old: list[float | None],
        new: list[float | None],
    ) -> float:
        """Use the original fixed-point metric without dictionary allocations."""
        max_change = 0.0
        for old_value, new_value in zip(old, new):
            if new_value is None:
                continue
            if old_value is None:
                max_change = max(max_change, abs(new_value))
                continue
            scale = max(abs(new_value), 1.0)
            max_change = max(max_change, abs(new_value - old_value) / scale)
        return max_change

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
