"""Derived-state evaluation for steady-state solves.

A FullFlow component can have two kinds of outputs:

* **Solver variables** controlled by SciPy's nonlinear solver.
* **Derived states** updated by each component's ``evaluate_states()`` method.

During every residual call the nonlinear solver tries a new vector of solver
variables.  The derived states then need to be recomputed without letting
component code accidentally overwrite those trial values.

``StateEvaluator`` also makes component order much less fragile.  If a component
tries to read an output that has not been produced yet in the current pass, the
evaluator postpones that component and tries it again after the remaining
components have had a chance to run.  This keeps simple explicit components such
as heat-transfer calculators readable: they do not need fake initial output
values just to satisfy construction order.
"""

from __future__ import annotations

from collections.abc import Callable

from .runtime import RuntimeCache


def _is_unassigned_state_error(error: BaseException) -> bool:
    """Return ``True`` when an exception was caused by an unassigned State.

    This is intentionally narrow.  Physical/modeling errors such as negative
    areas, invalid coefficients, division by zero, or custom component mistakes
    should still fail immediately.  Only the common order-of-evaluation case is
    deferred.
    """
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "has no assigned value" in str(current):
            return True
        current = current.__cause__ or current.__context__

    return False


def _callable_name(evaluate_states: Callable[[], None]) -> str:
    owner = getattr(evaluate_states, "__self__", None)
    if owner is not None:
        return f"{getattr(owner, 'name', type(owner).__name__)} ({type(owner).__name__})"
    return getattr(evaluate_states, "__name__", repr(evaluate_states))


class StateEvaluator:
    """Repeatedly call component ``evaluate_states()`` methods until settled.

    Parameters
    ----------
    cache_getter:
        Callable returning the current :class:`RuntimeCache`. The getter lets
        the evaluator survive model changes because it can always request the
        latest cached network view.
    """

    def __init__(self, cache_getter: Callable[[], RuntimeCache]) -> None:
        self._cache_getter = cache_getter

    def run(self, max_passes: int = 20, tolerance: float = 1e-10) -> None:
        """Evaluate all component-derived states.

        Components are evaluated repeatedly until derived State values stop
        changing.  If a component depends on another component's output but is
        listed earlier in the network, it is deferred for the current pass and
        retried later.  This preserves the simple component authoring model:

            evaluate_states() reads inputs and writes outputs

        without forcing users to manually order every component perfectly.
        """
        cache = self._cache_getter()
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

            # Retry only the components that were not ready yet.  On the next
            # pass, already-evaluated components will still be revisited after a
            # successful pending pass because ``all_callables`` is restored once
            # everything has run at least once.
            if pending:
                if evaluated_count == 0:
                    break
                all_callables = pending + [c for c in cache.evaluate_state_callables if c not in pending]
            else:
                all_callables = list(cache.evaluate_state_callables)

        if last_deferred_errors:
            lines = [
                "Component evaluation did not settle because one or more components still referenced unassigned States.",
                "",
                "Deferred components:",
            ]
            for evaluate_states, error in last_deferred_errors.items():
                lines.append(f"  - {_callable_name(evaluate_states)}: {str(error).splitlines()[0]}")
            lines.extend(
                [
                    "",
                    "Likely fixes:",
                    "  - Give the missing input State an initial value",
                    "  - Connect it to a component that computes it",
                    "  - Make it a dynamics or balances solve variable",
                    "  - Check for an accidental circular dependency with no initial guess",
                ]
            )
            raise RuntimeError("\n".join(lines)) from None
