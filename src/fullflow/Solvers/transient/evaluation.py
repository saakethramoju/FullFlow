"""Derived-state evaluation for transient solves.

The transient least-squares solver repeatedly tries new-time values for the
unknown vector.  Component ``evaluate_states()`` methods then update explicit
algebraic outputs and derivative quantities.  This evaluator mirrors the
steady-state evaluator: it repeatedly evaluates components until derived states
settle, while restoring the current unknown-vector guess after each component so
component side effects cannot overwrite solver variables.
"""

from __future__ import annotations

from collections.abc import Callable

from .runtime import TransientRuntimeCache
from fullflow.Exceptions import FullFlowConfigurationError, UnassignedStateError


def _is_unassigned_state_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, UnassignedStateError):
            return True
        current = current.__cause__ or current.__context__

    return False


def _callable_name(evaluate_states: Callable[[], None]) -> str:
    owner = getattr(evaluate_states, "__self__", None)
    if owner is not None:
        return f"{getattr(owner, 'name', type(owner).__name__)} ({type(owner).__name__})"
    return getattr(evaluate_states, "__name__", repr(evaluate_states))


class TransientStateEvaluator:
    """Repeatedly evaluate component-derived states during a timestep solve."""

    def __init__(self, cache_getter: Callable[[], TransientRuntimeCache]) -> None:
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self._cache_getter = cache_getter

    def run(
        self,
        max_passes: int = 5,
        tolerance: float = 1e-10,
        cache: TransientRuntimeCache | None = None,
    ) -> None:
        """Evaluate components until non-unknown states settle.

        Components that are temporarily waiting on another component's output
        are deferred and retried.  Real modeling errors still raise immediately.
        """
        cache = cache or self._cache_getter()
        iteration_snapshot = cache.snapshot_iteration_variables()
        all_callables = list(cache.evaluate_state_callables)
        last_deferred_errors: dict[Callable[[], None], BaseException] = {}

        for _ in range(max_passes):
            old_values = cache.collect_state_values()
            pending: list[Callable[[], None]] = []
            evaluated_count = 0

            for evaluate_states in all_callables:
                cache.restore_iteration_variables(iteration_snapshot)
                try:
                    evaluate_states()
                except Exception as error:
                    cache.restore_iteration_variables(iteration_snapshot)
                    if _is_unassigned_state_error(error):
                        pending.append(evaluate_states)
                        last_deferred_errors[evaluate_states] = error
                        continue
                    raise
                finally:
                    cache.restore_iteration_variables(iteration_snapshot)

                evaluated_count += 1

            new_values = cache.collect_state_values()
            if not pending and cache.max_state_change(old_values, new_values) < tolerance:
                return

            if pending:
                if evaluated_count == 0:
                    break
                all_callables = pending + [c for c in cache.evaluate_state_callables if c not in pending]
            else:
                all_callables = list(cache.evaluate_state_callables)

        if last_deferred_errors:
            lines = [
                "Transient component evaluation did not settle because one or more components still referenced unassigned States.",
                "",
                "Deferred components:",
            ]
            for evaluate_states, error in last_deferred_errors.items():
                lines.append(f"  - {_callable_name(evaluate_states)}: {str(error).splitlines()[0]}")
            raise FullFlowConfigurationError("\n".join(lines)) from None
