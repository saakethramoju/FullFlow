"""Derived-state evaluation for transient solves.

The transient least-squares solver repeatedly proposes new-time values for the
unknown vector.  Component ``evaluate_states()`` methods then update explicit
algebraic outputs and derivative States.  This evaluator mirrors the
steady-state evaluator: it repeatedly evaluates components until derived states
settle, while restoring SciPy's current unknown-vector guess after each
component so component side effects cannot overwrite solver variables.
"""

from __future__ import annotations

from collections.abc import Callable

from .runtime import TransientRuntimeCache


class TransientStateEvaluator:
    """Repeatedly evaluate component-derived states during a timestep solve."""

    def __init__(self, cache_getter: Callable[[], TransientRuntimeCache]) -> None:
        self._cache_getter = cache_getter

    def run(
        self,
        max_passes: int = 5,
        tolerance: float = 1e-10,
        cache: TransientRuntimeCache | None = None,
    ) -> None:
        """Evaluate components until non-unknown states settle."""
        cache = cache or self._cache_getter()
        iteration_snapshot = cache.snapshot_iteration_variables()
        evaluate_state_callables = cache.evaluate_state_callables

        for _ in range(max_passes):
            old_values = cache.collect_state_values()

            for evaluate_states in evaluate_state_callables:
                cache.restore_iteration_variables(iteration_snapshot)
                evaluate_states()
                cache.restore_iteration_variables(iteration_snapshot)

            new_values = cache.collect_state_values()
            if cache.max_state_change(old_values, new_values) < tolerance:
                return
