"""One-shot steady-state operations.

This file contains the executable pieces that the public ``SteadyState`` wrapper
calls after model options have been selected:

``StaticEvaluation``
    Runs pre-evaluation and derived-state propagation without nonlinear solving.
``NonlinearSolve``
    Builds the iteration vector, calls SciPy ``least_squares``, checks the final
    residual, and writes the solution back into the network.

The classes here intentionally do not know about model fallback, printing, or
user-facing argument names. That keeps numerical behavior isolated and easier to
debug.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import time

import numpy as np
from scipy.optimize import Bounds, least_squares

from .settings import LeastSquaresSettings, StateEvaluationSettings
from .statistics import SolverStatistics, statistics_path
from fullflow.Exports.HDF5 import HDF5Target, safe_group_name, write_tables


@dataclass(slots=True)
class StaticDiagnostics:
    """Timing information for a static evaluation."""

    elapsed_time: float


@dataclass(slots=True)
class SolveDiagnostics:
    """Data needed for verbose nonlinear-solver reporting."""

    sol: Any
    x0: np.ndarray
    settings: LeastSquaresSettings
    overconstrained: bool
    elapsed_time: float


class StaticEvaluation:
    """Evaluate a network without changing any iteration variable by solving.

    Static evaluation is useful for networks whose outputs are fully determined
    by assigned inputs and component ``evaluate_states()`` methods. It also acts
    as the degenerate solve path when a network has no iteration variables and
    no residuals.
    """

    def __init__(self, network, refresh_cache: Callable[[], Any], evaluator) -> None:
        self.network = network
        self.refresh_cache = refresh_cache
        self.evaluator = evaluator

    def run_once(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        state_settings: StateEvaluationSettings | None = None,
    ):
        """Run pre-evaluation, settle derived states, and export results."""
        state_settings = state_settings or StateEvaluationSettings()
        state_settings.validate()

        start_time = time.perf_counter()

        # Pre-evaluation can create/remove components through models or update
        # cached callables, so rebuild the runtime cache afterward.
        cache = self.refresh_cache()
        cache.run_pre_evaluation()
        self.refresh_cache()

        self.evaluator.run(
            max_passes=state_settings.max_passes,
            tolerance=state_settings.tolerance,
        )
        elapsed_time = time.perf_counter() - start_time

        records = self.network.save(filename=filename, return_type=return_type)

        if filename is not None:
            write_tables(
                HDF5Target(filename, f"{safe_group_name(self.network.name)}/steady_state"),
                {
                    "diagnostics": [
                        {
                            "solver_type": "static_evaluation",
                            "solve_time_s": elapsed_time,
                        }
                    ]
                },
            )

        return records, StaticDiagnostics(elapsed_time=elapsed_time)


class NonlinearSolve:
    """Execute one bounded or unbounded nonlinear least-squares solve."""

    def __init__(
        self,
        network,
        refresh_cache: Callable[[], Any],
        evaluator,
        residual_function: Callable[[np.ndarray], np.ndarray],
        static_runner: StaticEvaluation,
    ) -> None:
        self.network = network
        self.refresh_cache = refresh_cache
        self.evaluator = evaluator
        self.residual_function = residual_function
        self.static_runner = static_runner

    def run_once(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        least_squares_settings: LeastSquaresSettings | None = None,
        state_settings: StateEvaluationSettings | None = None,
        statistics: SolverStatistics | None = None,
        statistics_filename: str | None = None,
    ):
        """Run one steady-state solve on the current concrete network.

        The model system is handled one layer above this class. By the time this
        method runs, ``network.component_list`` already contains the components
        for the active model options.
        """
        ls_settings = least_squares_settings or LeastSquaresSettings()
        state_settings = state_settings or StateEvaluationSettings(max_passes=5)
        statistics = statistics or SolverStatistics(enabled=False)
        ls_settings.validate()
        state_settings.validate()

        # Let components prepare themselves, then rebuild the runtime view in
        # case those hooks changed network structure or iteration metadata.
        cache = self.refresh_cache()
        cache.run_pre_evaluation()
        cache = self.refresh_cache()

        x0 = np.array(cache.iteration_values, dtype=float)
        statistics.reset()
        statistics.configure(
            cache=cache,
            least_squares_settings=ls_settings,
            state_settings=state_settings,
            x0=x0,
        )

        # Compute derived states and the initial residual once before calling
        # SciPy. This catches bad setup early and lets us detect static networks.
        self.evaluator.run(
            max_passes=state_settings.max_passes,
            tolerance=state_settings.tolerance,
        )
        r0 = cache.collect_residuals()
        statistics.record(x0, r0, cache, phase="initial")

        # Seed the residual callback's invalid-trial history with the known-good
        # initial point.  This lets temporary invalid thermodynamic trial points
        # inside SciPy's finite-difference/Jacobian search receive a large
        # residual penalty instead of aborting the solve.
        residual_owner = getattr(self.residual_function, "__self__", None)
        if residual_owner is not None:
            if hasattr(residual_owner, "_last_valid_x"):
                residual_owner._last_valid_x = np.array(x0, dtype=float)
            if hasattr(residual_owner, "_last_valid_residual"):
                residual_owner._last_valid_residual = np.array(r0, dtype=float)

        if len(x0) == 0 and len(r0) == 0:
            return self.static_runner.run_once(
                filename=filename,
                return_type=return_type,
                state_settings=state_settings,
            )

        if len(r0) < len(x0):
            raise ValueError(
                "SteadyState solve requires at least as many residuals as iteration variables. "
                f"Got {len(x0)} iteration variables and {len(r0)} residuals."
            )

        solver_kwargs = self._least_squares_kwargs(cache, x0, ls_settings)

        start_time = time.perf_counter()
        try:
            sol = least_squares(**solver_kwargs)
        except Exception:
            statistics.export(statistics_filename)
            raise
        elapsed_time = time.perf_counter() - start_time

        statistics.finalize(
            sol=sol,
            elapsed_time=elapsed_time,
            cache=cache,
            least_squares_settings=ls_settings,
            state_settings=state_settings,
            overconstrained=len(r0) > len(x0),
        )

        try:
            self._check_solution(sol, ls_settings, cache, statistics)
        except Exception:
            statistics.export(statistics_filename)
            raise

        # ``least_squares`` does not automatically update the user's States.
        # Push the accepted solution back into the network, then recompute all
        # derived outputs one final time so saved/printed results are current.
        cache.assign_iteration_values(sol.x)
        self.evaluator.run(
            max_passes=state_settings.max_passes,
            tolerance=state_settings.tolerance,
        )
        records = self.network.save(filename=filename, return_type=return_type)

        if filename is not None:
            write_tables(
                HDF5Target(filename, f"{safe_group_name(self.network.name)}/steady_state"),
                {
                    "diagnostics": [
                        {
                            "solver_type": "steady_state",
                            "success": bool(getattr(sol, "success", False)),
                            "status": getattr(sol, "status", None),
                            "message": str(getattr(sol, "message", "")),
                            "function_evaluations": getattr(sol, "nfev", None),
                            "jacobian_evaluations": getattr(sol, "njev", None),
                            "cost": getattr(sol, "cost", None),
                            "optimality": getattr(sol, "optimality", None),
                            "max_abs_residual": float(np.max(np.abs(cache.collect_residuals()))) if len(cache.collect_residuals()) else 0.0,
                            "residual_count": len(cache.collect_residuals()),
                            "variable_count": len(sol.x),
                            "solve_time_s": elapsed_time,
                        }
                    ]
                },
            )

        statistics.export(statistics_filename)

        diagnostics = SolveDiagnostics(
            sol=sol,
            x0=x0,
            settings=ls_settings,
            overconstrained=len(r0) > len(x0),
            elapsed_time=elapsed_time,
        )
        return records, diagnostics

    def _least_squares_kwargs(
        self,
        cache,
        x0: np.ndarray,
        settings: LeastSquaresSettings,
    ) -> dict[str, Any]:
        """Build SciPy keyword arguments from current States and settings."""
        kwargs: dict[str, Any] = {
            "fun": self.residual_function,
            "x0": x0,
            "method": settings.solver_method,
            "x_scale": "jac",
            "jac": settings.jacobian_method,
            "ftol": settings.ftol,
            "xtol": settings.xtol,
            "gtol": settings.gtol,
        }

        if settings.solver_method == "lm":
            if any(state.has_bounds for state in cache.iteration_variables):
                raise ValueError("solver_method='lm' does not support bounded States.")
            return kwargs

        kwargs["bounds"] = Bounds(
            cache.lower_bounds,
            cache.upper_bounds,
            cache.keep_feasible,
        )
        return kwargs

    @staticmethod
    def _check_solution(
        sol: Any,
        settings: LeastSquaresSettings,
        cache,
        statistics: SolverStatistics,
    ) -> None:
        """Raise if SciPy failed or the final residual is too large."""
        final_residual = np.array(sol.fun, dtype=float)
        max_residual = np.max(np.abs(final_residual)) if len(final_residual) else 0.0

        if sol.success and max_residual <= settings.rtol:
            return

        try:
            residual_labels = cache.collect_residual_labels()
        except Exception:
            residual_labels = []

        if len(residual_labels) < len(final_residual):
            residual_labels.extend(
                f"residual[{i}]"
                for i in range(len(residual_labels), len(final_residual))
            )

        worst_rows = sorted(
            zip(residual_labels, final_residual),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )[:10]

        lines = [
            "Steady-state solve failed or converged to unacceptable residuals.",
            f"success = {sol.success}",
            f"message = {sol.message}",
            f"max |residual| = {max_residual:.3e}",
            f"residual tolerance = {settings.rtol:.3e}",
        ]

        if worst_rows:
            lines.extend(["", "Largest residuals:"])
            lines.extend(
                f"  - {label}: {value:.6e}"
                for label, value in worst_rows
            )

        statistics.print_failure_report(residual=final_residual, residual_labels=residual_labels)
        raise RuntimeError("\n".join(lines))
