"""Derived-state evaluation for steady-state solves."""

from __future__ import annotations

from collections.abc import Callable

from .runtime import RuntimeCache
from fullflow.Exceptions import FullFlowConfigurationError, UnassignedStateError


def _is_unassigned_state_error(error: BaseException) -> bool:
    """Return ``True`` when an exception was caused by an unassigned State."""
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, UnassignedStateError):
            return True
        current = current.__cause__ or current.__context__

    return False


def _callable_name(evaluate_states: Callable[[], None]) -> str:
    """Return a readable name for a callable."""
    owner = getattr(evaluate_states, "__self__", None)
    if owner is not None:
        return f"{getattr(owner, 'name', type(owner).__name__)} ({type(owner).__name__})"
    return getattr(evaluate_states, "__name__", repr(evaluate_states))


class StateEvaluator:
    """Repeatedly call component ``evaluate_states()`` methods until settled."""

    def __init__(self, cache_getter: Callable[[], RuntimeCache]) -> None:
        """Initialize the object and register any FullFlow state wiring."""
        self._cache_getter = cache_getter

    def run(self, max_passes: int = 20, tolerance: float = 1e-10) -> None:
        """Run derived-state evaluation to convergence."""
        cache = self._cache_getter()
        iteration_snapshot = cache.snapshot_iteration_variables()
        callables = cache.evaluate_state_callables
        all_callables = callables
        deferred_errors: dict[Callable[[], None], BaseException] = {}
        old_values = cache.collect_state_values_compact()
        new_values: list[float | None] = [None] * len(old_values)
        max_change = float("inf")

        for _ in range(max_passes):
            pending = []
            completed = 0
            for evaluate_states in all_callables:
                try:
                    evaluate_states()
                except Exception as error:
                    if _is_unassigned_state_error(error):
                        pending.append(evaluate_states)
                        deferred_errors[evaluate_states] = error
                        continue
                    raise
                finally:
                    cache.restore_iteration_variables(iteration_snapshot)
                completed += 1
                deferred_errors.pop(evaluate_states, None)

            cache.collect_state_values_compact(new_values)
            max_change = cache.max_state_change_compact(old_values, new_values)
            if not pending and max_change < tolerance:
                return
            old_values, new_values = new_values, old_values
            if pending:
                if completed == 0:
                    break
                pending_set = set(pending)
                all_callables = tuple(pending) + tuple(c for c in callables if c not in pending_set)
            else:
                all_callables = callables

        if deferred_errors:
            lines = ["Component evaluation could not resolve unassigned States:"]
            for evaluate_states, error in deferred_errors.items():
                lines.append(f"  - {_callable_name(evaluate_states)}: {str(error).splitlines()[0]}")
            raise FullFlowConfigurationError("\n".join(lines))
        raise FullFlowConfigurationError(
            f"Steady-state derived states did not settle after {max_passes} passes "
            f"(change={max_change:.3e}, tolerance={tolerance:.3e})."
        )
