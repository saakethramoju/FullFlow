"""Shared interpretation of FullFlow component equations.

The public component API has two solver-facing concepts:

``dynamics``
    Real time derivatives such as mass storage, energy storage, flow inertia,
    rotor inertia, or solid thermal capacitance.  Steady-state solves drive
    these derivatives to zero.  Transient solves integrate them.

``balances``
    Algebraic equations with no time storage of their own, such as a pump
    pressure match, a map inversion, or a user target.  Steady-state and
    transient solves both drive these residuals to zero.

There is intentionally no legacy component API here.  Components should expose
only ``evaluate_states()``, ``dynamics``, and ``balances``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fullflow.System.Component import Component
from fullflow.System.State import State, is_assignable_state_like, is_state_like


@dataclass(frozen=True, slots=True)
class DynamicEquation:
    """One dynamic equation from a component.

    ``variable`` is the State varied by the nonlinear solver.  ``state`` is the
    State integrated by the transient solver.  They are normally the same State,
    but volumes can solve with pressure/enthalpy while integrating mass/energy.
    """

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
    raise TypeError(f"{property_name} must return a list or tuple, not {type(value).__name__}.")




def _is_unassigned_state_error(error: BaseException) -> bool:
    """Return True when an exception was caused by an unassigned State.

    This helper is used only during equation discovery. It lets the solver
    evaluate components once before reading ``dynamics``/``balances`` so
    derivative/error attributes can be created inside ``evaluate_states()``
    instead of being pre-initialized in constructors.
    """
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "has no assigned value" in str(current):
            return True
        current = current.__cause__ or current.__context__

    return False


def _callable_name(method: Any) -> str:
    owner = getattr(method, "__self__", None)
    if owner is not None:
        return f"{getattr(owner, 'name', type(owner).__name__)} ({type(owner).__name__})"
    return getattr(method, "__name__", repr(method))


def evaluate_components_for_equation_discovery(components: list[Any] | tuple[Any, ...], *, max_passes: int = 20) -> None:
    """Evaluate components before reading their equation properties.

    Component authors should be able to write clean constructors that mostly do
    only ``self.setup()``. Derivative and balance-error attributes such as
    ``mass_flow_dot`` or ``pressure_error`` can be created in
    ``evaluate_states()``. Runtime caches call this helper before first reading
    ``dynamics`` and ``balances`` so those attributes exist.

    Only order-of-evaluation failures caused by unassigned upstream States are
    deferred. Physical/modeling errors still raise immediately.
    """
    pending = list(components)
    last_deferred_errors: dict[int, tuple[Any, BaseException]] = {}

    for _ in range(max_passes):
        for component in components:
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
            for component in components
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
        raise RuntimeError("\n".join(lines)) from None

def component_dynamics(owner: Any) -> list[DynamicEquation]:
    """Return normalized dynamic equations for a component."""
    raw_equations = _as_list(owner.dynamics, property_name="dynamics") if _custom_component_property(owner, "dynamics") else []

    equations: list[DynamicEquation] = []
    for index, entry in enumerate(raw_equations):
        if not isinstance(entry, tuple):
            raise TypeError(f"{owner.name}: dynamics[{index}] must be a tuple.")

        if len(entry) == 2:
            variable, derivative = entry
            state = variable
        elif len(entry) == 3:
            variable, state, derivative = entry
        else:
            raise ValueError(
                f"{owner.name}: dynamics[{index}] must be either "
                "(variable, derivative) or (variable, state, derivative)."
            )

        if not is_assignable_state_like(variable):
            raise TypeError(f"{owner.name}: dynamics[{index}] variable must be an assignable State.")
        if not is_state_like(state):
            raise TypeError(f"{owner.name}: dynamics[{index}] integrated state must be a State.")

        equations.append(DynamicEquation(owner=owner, variable=variable, state=state, derivative=derivative, index=index))

    return equations


def component_balances(owner: Any) -> list[BalanceEquation]:
    """Return normalized algebraic balances for a component."""
    raw_equations = _as_list(owner.balances, property_name="balances") if _custom_component_property(owner, "balances") else []

    equations: list[BalanceEquation] = []
    for index, entry in enumerate(raw_equations):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(f"{owner.name}: balances[{index}] must be (variable, residual).")

        variable, residual = entry
        if not is_assignable_state_like(variable):
            raise TypeError(f"{owner.name}: balances[{index}] variable must be an assignable State.")

        equations.append(BalanceEquation(owner=owner, variable=variable, residual=residual, index=index))

    return equations


def balance_object_equations(owner: Any) -> list[BalanceEquation]:
    """Return the normalized equation for a user ``Balance`` object.

    The residual is kept as a callable instead of being evaluated during runtime
    cache setup.  It is evaluated only after the network has run
    ``evaluate_states()`` for the current solver guess.
    """
    return [BalanceEquation(owner=owner, variable=owner.variable, residual=owner._residual, index=0)]
