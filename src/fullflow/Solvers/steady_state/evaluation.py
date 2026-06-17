"""Derived-state evaluation for steady-state solves.

A FullFlow component can have two kinds of outputs:

* **Iteration variables** controlled by SciPy's nonlinear solver.
* **Derived states** updated by each component's ``evaluate_states()`` method.

During every residual call the nonlinear solver proposes a new vector of
iteration variables. The derived states then need to be recomputed without
letting component code accidentally overwrite those proposed iteration values.
``StateEvaluator`` handles that fixed-point pass.
"""

from __future__ import annotations

from collections.abc import Callable

from .runtime import RuntimeCache


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

        ``evaluate_states()`` is run in network order. After each component is
        evaluated, the iteration variables are restored to the solver-proposed
        snapshot. This protects the nonlinear solve from component side effects
        while still allowing normal derived-state propagation.
        """
        cache = self._cache_getter()
        iteration_snapshot = cache.snapshot_iteration_variables()

        for _ in range(max_passes):
            old_values = cache.collect_state_values()

            for evaluate_states in cache.evaluate_state_callables:
                cache.restore_iteration_variables(iteration_snapshot)
                evaluate_states()
                cache.restore_iteration_variables(iteration_snapshot)

            new_values = cache.collect_state_values()
            if cache.max_state_change(old_values, new_values) < tolerance:
                return
