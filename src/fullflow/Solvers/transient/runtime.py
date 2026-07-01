"""Runtime network view used by the transient solver.

``Network`` stays a lightweight container.  This module is the transient
solver's interpretation of that container: which component ``dynamics`` are
integrated, which component/user ``balances`` are algebraic unknowns, and which
State bounds should be passed to SciPy.

The public component API is intentionally small:

* ``evaluate_states()`` updates outputs, derivatives, and balance errors.
* ``dynamics`` lists real storage/inertia/capacitance equations.
* ``balances`` lists algebraic equations with no storage.

Steady-state solves drive dynamic derivatives and algebraic balances to zero.
Transient solves integrate dynamic equations and close algebraic balances at
each new timestep.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import copy
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
from fullflow.Exceptions import SolverSetupError
from fullflow.Solvers.equations import (
    balance_object_equations,
    component_balances,
    component_dynamics,
    evaluate_components_for_equation_discovery,
)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class TransientItem:
    """One dynamic equation owned by one component.

    ``variable`` is the State controlled by SciPy in the new-time nonlinear
    solve.  ``state`` is the conserved/integrated State used in the backward
    Euler residual.  For most components these are the same object.  For a
    fluid volume they can be different: pressure/enthalpy can be the solver
    variables while mass/total-internal-energy are the integrated states.

    The derivative value is intentionally **not** stored here.  Instead, the
    cache stores the derivative list index.  During every residual call, after
    the network has been evaluated at SciPy's current new-time guess, the
    solver re-reads ``owner.dynamics[derivative_index]``.  That
    keeps direct expression derivatives current.
    """

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
    """Cached solver metadata for one transient ``Network`` instance.

    The cache is rebuilt whenever ``network.version`` changes.  Runtime metadata
    lives solver-side so transient rules do not complicate the base ``Network``
    class.
    """

    def __init__(
        self,
        network,
        ignore_balances=None,
        force_steady=None,
        force_steady_exceptions=None,
    ) -> None:
        self.network = network
        self.ignore_balances = ignore_balances
        self.force_steady = force_steady
        self.force_steady_exceptions = force_steady_exceptions
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
        """Rebuild transient items, algebraic items, callables, and bounds.

        Phase 6 keeps the expensive network walk here.  Residual calls should
        only assign SciPy's vector, evaluate cached component callables, and
        collect cached equation blocks.  Nothing sequence-related lives in this
        cache.
        """
        self.component_list = tuple(self.network.component_list)
        self.balance_list, self.ignored_balance_list = filter_user_balances(
            self.network.balance_list,
            self.ignore_balances,
        )
        self.model_list = tuple(self.network.model_list)

        # Component equation properties are allowed to reference derivative or
        # balance-error attributes created inside evaluate_states().  Evaluate
        # once before reading dynamics/balances so component constructors do not
        # need boilerplate like self.mass_flow_dot = 0.0.
        evaluate_components_for_equation_discovery(self.component_list)

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

        self.algebraic_residual_owners = self.component_list
        self.balance_residual_owners = self.balance_list

        # These references are used only to decide whether repeated
        # evaluate_states() passes have settled.  They exclude solver unknowns
        # so the evaluator never treats SciPy's current guess as a derived state.
        self.state_refs = self._collect_state_refs()

        # These references are used for failed-step rollback.  They include
        # solver unknowns, explicit outputs, hidden mode states, lookup backing
        # states, and network time so a rejected timestep cannot leave partial
        # residual-call mutations behind.
        self.all_state_refs = self._collect_all_state_refs()
        self.version = self.network.version

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
        return self._transient_variable_lists[id(component)]

    def _component_transient_states(self, component: Any) -> list[Any]:
        return self._transient_state_lists[id(component)]

    def _component_transient_derivatives(self, component: Any) -> list[Any]:
        # Derivative values can depend on the current SciPy trial point, so only
        # variable/state membership is cached.  Derivatives are re-read from
        # component.dynamics after evaluate_states() on every residual call.
        return [equation.derivative for equation in component_dynamics(component)]

    def _component_is_dynamic(self, component: Any) -> bool:
        return id(component) in self.dynamic_component_ids

    @staticmethod
    def _is_valid_derivative(value: Any) -> bool:
        """Return True for supported derivative objects.

        A derivative may be a State or a plain float.  Booleans are
        rejected even though ``bool`` subclasses ``int`` because they are
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
        raise SolverSetupError(
            f"{component.name}: dynamics derivative[{index}] must be a State "
            f"or a numeric value. Got {type(derivative).__name__}."
        )

    def _collect_transient_items(self) -> list[TransientItem]:
        """Collect dynamic unknowns, integrated states, and derivatives.

        These come from component ``dynamics`` tuples.  The two-entry form
        ``(state, derivative)`` integrates the same State that the solver varies.
        The three-entry form ``(solve_variable, integrated_state, derivative)``
        lets a component solve with a convenient variable such as pressure while
        integrating a conservative quantity such as mass.
        """
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
        """Collect variables from component algebraic balances.

        Dynamic equations are integrated.  Algebraic balances, if a component
        actually needs them, are closed at the same new-time point.
        """
        items: list[IterationItem] = []

        for component in self.component_list:
            for equation in component_balances(component):
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
        """Return solver unknowns with shared State objects kept only once.

        Shared States are intentional in transient tank models.  For example, two
        Volume components may share one tank pressure, and a variable volume may
        appear both as a Volume transient algebraic variable and as the variable
        of a Balance that supplies the geometry equation.
        """
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
        """Reject multiple integration equations for the same dynamic integrated state.

        Different solver variables may be used to close different conservation
        equations, but each integrated State should have one and only one
        transient derivative.  Sharing a dynamic integrated state across two integration
        residuals would advance the same history value twice and is almost
        always a modeling error.
        """
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
        return f"{owner.name}:dynamics[{index}]"

    def _current_derivative(self, item: TransientItem) -> Any:
        """Read the derivative value/property for ``item`` at the current state."""
        derivatives = self._component_transient_derivatives(item.owner)

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
            equations = component_balances(owner)
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
        """Capture all reachable mutable State values before a timestep attempt.

        SciPy residual calls can touch more than the explicit solver unknowns.
        Explicit components may write mass flow, heat flow, lookup guesses, mode
        States, or other derived outputs before a timestep is accepted.  A failed
        timestep must restore those values before the outer loop retries with a
        smaller dt.
        """
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
        """Restore a snapshot created by :meth:`snapshot_iteration_variables`."""
        for state, value in snapshot:
            state.value = value

    def run_pre_evaluation(self) -> None:
        """Run all component ``pre_evaluation()`` hooks in network order."""
        for pre_evaluation in self.pre_evaluation_callables:
            pre_evaluation()

    def set_transient_context(self, *, dt: float) -> None:
        """Pass timestep context to every component before evaluation."""
        for component in self.component_list:
            component.set_transient_context(dt=dt)

    def _states_with_history(self) -> tuple[Any, ...]:
        """States whose accepted previous values should be available next step.

        Dynamic integrated States are required for backward Euler.  Other
        reachable States are stored too so components such as variable-volume
        ``Volume`` can compute simple accepted-step differences like dV/dt
        without exposing any special public API hook.
        """
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
        for state in self._states_with_history():
            state.store_previous()

    def clear_previous_values(self) -> None:
        """Clear stored accepted-step history."""
        for state in self._states_with_history():
            state.clear_previous()

    def _derivative_value(self, item: TransientItem) -> float:
        """Return the current numeric derivative for ``item``.

        The derivative is re-read from the owning component every time this is
        called.  That is what makes direct expression properties behave like
        derivative States: both are evaluated at the current new-time guess.
        """
        return resolve_numeric(self._current_derivative(item))

    @staticmethod
    def flatten_residuals(residual_source: Any) -> list[float]:
        """Convert callable, State, or numeric residuals into floats."""
        if callable(residual_source):
            residual_source = residual_source()
        if residual_source is None:
            return []
        if not isinstance(residual_source, (list, tuple)):
            residual_source = (residual_source,)

        values: list[float] = []
        for value in residual_source:
            if callable(value):
                value = value()
            values.append(resolve_numeric(value))
        return values

    @staticmethod
    def _transient_scale(
        item: TransientItem,
        *,
        dt: float,
        derivative: float,
    ) -> float:
        """Scale used for internally generated integration residuals.

        Algebraic residuals are left exactly as components and balances return
        them, matching the steady-state solver.  Only backward-Euler
        integration residuals are normalized because their raw units can be kg,
        J, K, rad/s, etc.

        The scale includes the current state, the previous state, and the
        expected timestep change.  This prevents tiny absolute conservation
        defects, such as ``1e-8 kg/m^3`` on a ``1000 kg/m^3`` density state,
        from causing a failed timestep.
        """
        return max(
            abs(float(item.state.value)),
            abs(float(item.state.previous)),
            abs(float(dt * derivative)),
            1.0,
        )

    def collect_residuals(self, dt: float) -> np.ndarray:
        """Collect transient, algebraic, and balance residuals.

        Dynamic residuals are implicit backward-Euler residuals evaluated at the
        current guessed new-time state unless a component has been forced steady:

            dynamic mode:       state.value - state.previous - dt * derivative = 0
            force-steady mode:  derivative = 0

        Integration residuals are divided by a simple state scale so ``rtol`` has
        a useful per-timestep meaning across different state units.  Force-steady
        derivative residuals are left in their physical units, matching the
        steady-state solver's dynamic trim equations.
        """
        residuals: list[float] = []

        for item in self.transient_items:
            derivative = self._derivative_value(item)

            if item.force_steady:
                residuals.append(derivative)
                continue

            state = float(item.state.value)
            previous = float(item.state.previous)
            scale = self._transient_scale(item, dt=dt, derivative=derivative)
            residuals.append((state - previous - dt * derivative) / scale)

        for component in self.algebraic_residual_owners:
            for equation in component_balances(component):
                residuals.extend(self.flatten_residuals(equation.residual))

        for balance in self.balance_residual_owners:
            for equation in balance_object_equations(balance):
                residuals.extend(self.flatten_residuals(equation.residual))

        return np.array(residuals, dtype=float)

    def validate_residual_count(self, residuals: np.ndarray) -> None:
        """Fail early if there are fewer equations than unknowns.

        Overconstrained systems are allowed because least-squares can minimize
        more residuals than unknowns.  Underdetermined systems are almost always
        missing component residuals, a balance, or a transient derivative.
        """
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
            state_id = id(state)
            if state_id in seen:
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
