"""Shared interpretation of FullFlow component equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fullflow.System.Component import Component
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

def component_dynamics(owner: Any) -> list[DynamicEquation]:
    """Return normalized dynamic equations for a component."""
    raw_equations = _as_list(owner.dynamics, property_name="dynamics") if _custom_component_property(owner, "dynamics") else []

    equations: list[DynamicEquation] = []
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

        if not is_assignable_state_like(variable):
            raise FullFlowConfigurationError(f"{owner.name}: dynamics[{index}] variable must be an assignable State.")
        if not is_state_like(state):
            raise FullFlowConfigurationError(f"{owner.name}: dynamics[{index}] integrated state must be a State.")

        equations.append(DynamicEquation(owner=owner, variable=variable, state=state, derivative=derivative, index=index))

    return equations


def component_balances(owner: Any) -> list[BalanceEquation]:
    """Return normalized algebraic balances for a component."""
    raw_equations = _as_list(owner.balances, property_name="balances") if _custom_component_property(owner, "balances") else []

    equations: list[BalanceEquation] = []
    for index, entry in enumerate(raw_equations):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise FullFlowConfigurationError(f"{owner.name}: balances[{index}] must be (variable, residual).")

        variable, residual = entry
        if not is_assignable_state_like(variable):
            raise FullFlowConfigurationError(f"{owner.name}: balances[{index}] variable must be an assignable State.")

        equations.append(BalanceEquation(owner=owner, variable=variable, residual=residual, index=index))

    return equations


def balance_object_equations(owner: Any) -> list[BalanceEquation]:
    """Return the normalized equation for a user ``Balance`` object."""
    return [BalanceEquation(owner=owner, variable=owner.variable, residual=owner._residual, index=0)]
