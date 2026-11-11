"""Shared interpretation of FullFlow component equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fullflow.System.Component import Component
from fullflow.System.StateArray import StateArray
from fullflow.System.State import State, is_assignable_state_like, is_state_like
from fullflow.Exceptions import FullFlowConfigurationError, UnassignedStateError


@dataclass(frozen=True, slots=True)
class DynamicEquation:
    """One dynamic equation from a component."""

    owner: Any
    variable: Any
    state: Any
    derivative: Any
    index: int


@dataclass(frozen=True, slots=True)
class BalanceEquation:
    """One algebraic balance equation from a component or user Balance."""

    owner: Any
    variable: Any
    residual: Any
    index: int


def _custom_component_property(owner: Any, name: str) -> bool:
    """Return True only when a component class overrides a base property."""
    return getattr(type(owner), name, None) is not getattr(Component, name, None)


def _as_list(value: Any, *, property_name: str) -> list[Any]:
    """Normalize equation properties to ordinary lists with clear errors."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise FullFlowConfigurationError(f"{property_name} must return a list or tuple, not {type(value).__name__}.")


def _is_unassigned_state_error(error: BaseException) -> bool:
    """Return True when an exception was caused by an unassigned State."""
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, UnassignedStateError):
            return True
        current = current.__cause__ or current.__context__

    return False


def _callable_name(method: Any) -> str:
    """Return a readable name for a callable."""
    owner = getattr(method, "__self__", None)
    if owner is not None:
        return f"{getattr(owner, 'name', type(owner).__name__)} ({type(owner).__name__})"
    return getattr(method, "__name__", repr(method))


def evaluate_components_for_equation_discovery(
    components: list[Any] | tuple[Any, ...],
    *,
    max_passes: int = 20,
    solve_mode: str = "transient",
) -> None:
    """Evaluate components before reading their equation properties."""
    active_components = [
        component
        for component in components
        if component.active_in_solver(solve_mode)
    ]
    pending = list(active_components)
    last_deferred_errors: dict[int, tuple[Any, BaseException]] = {}

    for _ in range(max_passes):
        for component in active_components:
            component.pre_evaluation()

        next_pending: list[Any] = []
        evaluated_count = 0

        for component in pending:
            try:
                component.evaluate_states()
            except Exception as error:
                if _is_unassigned_state_error(error):
                    next_pending.append(component)
                    last_deferred_errors[id(component)] = (component, error)
                    continue
                raise

            evaluated_count += 1
            last_deferred_errors.pop(id(component), None)

        if not next_pending:
            return

        if evaluated_count == 0:
            break

        pending_ids = {id(component) for component in next_pending}
        pending = next_pending + [
            component
            for component in active_components
            if id(component) not in pending_ids
        ]

    if last_deferred_errors:
        lines = [
            "Could not discover component equations because one or more components still referenced unassigned States.",
            "",
            "Deferred components:",
        ]
        for component, error in last_deferred_errors.values():
            lines.append(f"  - {component.name} ({type(component).__name__}): {str(error).splitlines()[0]}")
        lines.extend([
            "",
            "Likely fixes:",
            "  - Give the missing input State an initial value",
            "  - Connect it to a component that computes it",
            "  - Make it a dynamics or balances solve variable with an initial guess",
        ])
        raise FullFlowConfigurationError("\n".join(lines)) from None


class _GroupLayouts:
    def __init__(self):
        self.entries = {}


class _VectorSource:
    def __init__(self, source, shape):
        self.shape = shape
        self.update(source)

    def update(self, source):
        if is_state_like(source):
            source = source.value
        elif callable(source):
            source = source()
            if is_state_like(source):
                source = source.value
        values = np.asarray(source)
        if values.shape != self.shape:
            raise FullFlowConfigurationError(
                f"Grouped equation requires shape {self.shape}; got {values.shape}."
            )
        if values.dtype.kind not in "biuf":
            raise FullFlowConfigurationError("Grouped equation requires numeric values.")
        self.current = values

    def values(self):
        return self.current


def _group_entries(owner, kind, index, variable, state, source):
    if state is not None and (not isinstance(state, StateArray) or state.shape != variable.shape):
        raise FullFlowConfigurationError("Grouped dynamics require matching StateArray variable and integrated-state shapes.")
    for member in variable._members.flat:
        _require_scalar_state(member)
        if member.is_derived:
            raise FullFlowConfigurationError("Grouped equation variables must be assignable scalar States.")
    cache = getattr(owner, "_fullflow_equation_layouts", None)
    if cache is None:
        cache = owner._fullflow_equation_layouts = _GroupLayouts()
    layout = cache.entries
    key = kind, index
    signature = (tuple(id(s) for s in variable._members.flat), variable.shape,
                 None if state is None else tuple(id(s) for s in state._members.flat))
    previous = layout.get(key)
    if previous is not None and previous[0] == signature:
        previous[1].update(source)
        return previous[2]
    vector = _VectorSource(source, variable.shape)
    entries = []
    for position in np.ndindex(variable.shape):
        residual = State._derived(lambda position=position: vector.values()[position])
        if state is None:
            entries.append((variable.at(position), residual))
        else:
            integrated = state.at(position)
            _require_scalar_state(integrated)
            entries.append((variable.at(position), integrated, residual))
    layout[key] = signature, vector, entries
    return entries


def validate_group_ownership(owners):
    """Reject overlapping declarations involving grouped scalar members."""
    counts = {}
    grouped = set()
    for owner in owners:
        for kind in ("dynamics", "balances"):
            for entry in (getattr(owner, kind, ()) or ()):
                if not isinstance(entry, tuple) or not entry:
                    continue
                variable = entry[0]
                members = variable._members.flat if isinstance(variable, StateArray) else (variable,)
                for member in members:
                    identity = id(member)
                    counts[identity] = counts.get(identity, 0) + 1
                    if isinstance(variable, StateArray):
                        grouped.add(identity)
    if any(counts[identity] > 1 for identity in grouped):
        raise FullFlowConfigurationError("Duplicate or overlapping grouped equation ownership for a scalar State.")


def _require_scalar_state(state: Any) -> None:
    if type(state) is State and state._expr is None:
        if state._array_shape is None or len(state._array_shape) == 0:
            return
    if isinstance(state, State) and state.is_assigned:
        try:
            value = state.value
        except UnassignedStateError:
            return
        if isinstance(value, np.ndarray) and value.ndim > 0:
            raise FullFlowConfigurationError(
                f"{state.label}: array-valued State cannot be a solver unknown or integrated state; "
                "use StateArray of scalar States."
            )


def _is_assignable_variable(variable: Any) -> bool:
    """Check exact States directly; retain the generic proxy validation path."""
    if type(variable) is State:
        if variable._array_shape is not None and len(variable._array_shape):
            _require_scalar_state(variable)
        return not variable.is_derived
    _require_scalar_state(variable)
    return is_assignable_state_like(variable)


def _dynamic_entries(owner: Any) -> list[tuple]:
    """Validate current dynamic entries without allocating equation wrappers."""
    raw_equations = _as_list(owner.dynamics, property_name="dynamics") if _custom_component_property(owner, "dynamics") else []

    entries = []
    for index, entry in enumerate(raw_equations):
        if not isinstance(entry, tuple):
            raise FullFlowConfigurationError(f"{owner.name}: dynamics[{index}] must be a tuple.")

        if len(entry) == 2:
            variable, derivative = entry
            state = variable
        elif len(entry) == 3:
            variable, state, derivative = entry
        else:
            raise FullFlowConfigurationError(
                f"{owner.name}: dynamics[{index}] must be either "
                "(variable, derivative) or (variable, state, derivative)."
            )

        if isinstance(variable, StateArray):
            entries.extend(_group_entries(owner, "dynamics", index, variable, state, derivative))
            continue
        if not _is_assignable_variable(variable):
            raise FullFlowConfigurationError(f"{owner.name}: dynamics[{index}] variable must be an assignable State.")
        if type(state) is not State and not is_state_like(state):
            raise FullFlowConfigurationError(f"{owner.name}: dynamics[{index}] integrated state must be a State.")

        _require_scalar_state(state)
        entries.append((variable, state, derivative))

    return entries


def component_dynamics(owner: Any) -> list[DynamicEquation]:
    """Return normalized dynamic equations for a component."""
    return [
        DynamicEquation(owner, variable, state, derivative, index)
        for index, (variable, state, derivative) in enumerate(_dynamic_entries(owner))
    ]


def _balance_entries(owner: Any) -> list[tuple]:
    """Validate current balance entries without allocating equation wrappers."""
    raw_equations = _as_list(owner.balances, property_name="balances") if _custom_component_property(owner, "balances") else []

    entries = []
    for index, entry in enumerate(raw_equations):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise FullFlowConfigurationError(f"{owner.name}: balances[{index}] must be (variable, residual).")

        variable, residual = entry
        if type(variable) is State:
            if variable._array_shape is not None and len(variable._array_shape):
                _require_scalar_state(variable)
            if variable.is_derived:
                raise FullFlowConfigurationError(f"{owner.name}: balances[{index}] variable must be an assignable State.")
            entries.append((variable, residual))
            continue
        if isinstance(variable, StateArray):
            entries.extend(_group_entries(owner, "balances", index, variable, None, residual))
            continue
        if not _is_assignable_variable(variable):
            raise FullFlowConfigurationError(f"{owner.name}: balances[{index}] variable must be an assignable State.")

        entries.append((variable, residual))

    return entries


def component_balances(owner: Any) -> list[BalanceEquation]:
    """Return normalized algebraic balances for a component."""
    return [
        BalanceEquation(owner, variable, residual, index)
        for index, (variable, residual) in enumerate(_balance_entries(owner))
    ]


def balance_object_equations(owner: Any) -> list[BalanceEquation]:
    """Return the normalized equation for a user ``Balance`` object."""
    if isinstance(owner.variable, StateArray):
        return [BalanceEquation(owner, variable, residual, index)
                for index, (variable, residual) in enumerate(
                    _group_entries(owner, "balances", 0, owner.variable, None, owner._residual))]
    _require_scalar_state(owner.variable)
    return [BalanceEquation(owner=owner, variable=owner.variable, residual=owner._residual, index=0)]
