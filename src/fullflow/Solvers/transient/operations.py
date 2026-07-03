"""Executable transient timestep operations.

This module contains the numerical core used by :class:`Transient`:

* initialize transient history at the starting time,
* build SciPy ``least_squares`` arguments from current State bounds,
* solve one implicit backward-Euler timestep,
* check the accepted residual against the per-step tolerance.

This module solves one requested timestep.  The public ``Transient.solve`` loop
can retry a failed step with smaller substeps; this inner object keeps the
single-step residual solve focused and deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import time

import numpy as np
from scipy.optimize import Bounds, least_squares

from fullflow.Solvers.steady_state.settings import LeastSquaresSettings, StateEvaluationSettings
from fullflow.Exceptions import SensorDataStop, SolverSetupError, TransientStepError


@dataclass(slots=True)
class StepDiagnostics:
    """Data recorded for one accepted implicit timestep.

    ``x0`` is the nonlinear initial guess for this timestep.  ``sol`` is the
    SciPy result object when a nonlinear solve was needed.  ``sol`` is ``None``
    for static transient steps with no unknowns.  ``residual`` is always the
    residual recomputed after the accepted values have been written back to the
    network and all component states have been evaluated one final time.
    """

    time: float
    dt: float
    x0: np.ndarray
    sol: Any | None
    residual: np.ndarray
    elapsed_time: float
    retries: int = 0

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residual))) if len(self.residual) else 0.0

    @property
    def rms_residual(self) -> float:
        return float(np.sqrt(np.mean(self.residual**2))) if len(self.residual) else 0.0


class TransientStepSolve:
    """Solve one implicit backward-Euler timestep with SciPy least_squares."""

    def __init__(
        self,
        network,
        cache_getter: Callable[[], Any],
        evaluator,
    ) -> None:
        self.network = network
        self._cache_getter = cache_getter
        self.evaluator = evaluator
        self._active_time: float = 0.0
        self._active_dt: float = 0.0
        self._active_state_settings = StateEvaluationSettings(max_passes=5)
        self._active_cache: Any | None = None
        self._last_debug_x: np.ndarray | None = None
        self._last_debug_residual: np.ndarray | None = None
        self._last_debug_error: Exception | None = None
        self._last_valid_x: np.ndarray | None = None
        self._last_valid_residual: np.ndarray | None = None
        self._invalid_residual_penalty = 1.0e12

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Residual function passed to SciPy for the current timestep.

        SciPy tries values for all new-time unknowns.  We write those values
        into the network, set the network clock to the target timestep time,
        evaluate all components, then collect residuals in this order:

        1. implicit transient integration residuals,
        2. algebraic component residuals,
        3. balance residuals.
        """
        cache = self._active_cache or self._cache_getter()
        self._last_debug_x = np.array(x, dtype=float)
        self._last_debug_residual = None
        self._last_debug_error = None

        cache.assign_iteration_values(x)
        self.network.time.value = self._active_time
        cache.set_transient_context(dt=self._active_dt)

        try:
            self.evaluator.run(
                max_passes=self._active_state_settings.max_passes,
                tolerance=self._active_state_settings.tolerance,
                cache=cache,
            )
            residual = cache.collect_residuals(self._active_dt)
            self._last_debug_residual = residual
            self._last_valid_x = np.array(x, dtype=float)
            self._last_valid_residual = np.array(residual, dtype=float)
            return residual

        except Exception as error:
            if isinstance(error, SensorDataStop):
                raise

            self._last_debug_error = error
            penalty_residual = self._invalid_trial_residual(x)

            if penalty_residual is not None:
                self._last_debug_residual = penalty_residual
                return penalty_residual

            try:
                self._last_debug_residual = cache.collect_residuals(self._active_dt)
            except Exception:
                self._last_debug_residual = None

            raise TransientStepError(
                "Transient solver encountered an error while evaluating the network "
                "inside a timestep residual call.\n\n"
                f"time = {self._active_time:.9g}\n"
                f"dt = {self._active_dt:.9g}\n"
                f"Original error:\n{type(error).__name__}: {error}"
            ) from error

    def _invalid_trial_residual(self, x: np.ndarray) -> np.ndarray | None:
        """Return a large residual for invalid nonlinear trial points."""
        if self._last_valid_residual is None:
            return None

        count = len(self._last_valid_residual)

        if count == 0:
            return np.array([], dtype=float)

        residual = np.full(count, self._invalid_residual_penalty, dtype=float)

        if self._last_valid_x is not None and len(x) > 0:
            dx = np.array(x, dtype=float) - self._last_valid_x
            scale = np.maximum(np.abs(self._last_valid_x), 1.0)
            normalized_dx = dx / scale

            for index in range(min(count, len(normalized_dx))):
                residual[index] += 1.0e6 * normalized_dx[index]

        return residual

    def initialize(self, state_settings: StateEvaluationSettings) -> None:
        """Evaluate the initial network and store transient ``previous`` values."""
        state_settings.validate()
        cache = self._cache_getter()
        cache.run_pre_evaluation()
        cache = self._cache_getter()
        self._active_cache = cache

        try:
            cache.set_transient_context(dt=0.0)
            self.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance,
                cache=cache,
            )
            cache.store_previous_values()
        finally:
            self._active_cache = None

    def run_once(
        self,
        *,
        t_new: float,
        dt: float,
        least_squares_settings: LeastSquaresSettings,
        state_settings: StateEvaluationSettings,
    ) -> StepDiagnostics:
        """Solve and accept one timestep.

        The component equations are evaluated directly during the nonlinear
        solve.  Component authors write ordinary component equations; timestep
        retry and residual acceptance are handled by the transient solver.
        """
        least_squares_settings.validate()
        state_settings.validate()
        self._active_time = float(t_new)
        self._active_dt = float(dt)
        self._active_state_settings = state_settings

        cache = self._cache_getter()
        self._active_cache = cache
        x0 = cache.iteration_value_array()
        accepted_snapshot = cache.snapshot_mutable_states()
        accepted_time = float(self.network.time.value)

        self._last_valid_x = None
        self._last_valid_residual = None

        try:
            # Evaluate the residual once before SciPy.  This catches invalid
            # initial states and detects the degenerate case where there is
            # nothing to solve.
            self.network.time.value = t_new
            r0 = self.residual(x0)
            cache.validate_residual_count(r0)

            if len(x0) == 0:
                # Static transient step: explicit components may still update
                # outputs as time changes, but no nonlinear solve
                # is needed.  If residuals exist, they must already satisfy the
                # per-step tolerance because there are no unknowns available to
                # change them.
                self._check_residual_vector(
                    residual=r0,
                    success=True,
                    message="No nonlinear variables; static transient evaluation.",
                    settings=least_squares_settings,
                    cache=cache,
                    t_new=t_new,
                    dt=dt,
                )
                self.evaluator.run(
                    max_passes=state_settings.max_passes,
                    tolerance=state_settings.tolerance,
                    cache=cache,
                )
                cache.store_previous_values()
                return StepDiagnostics(
                    time=t_new,
                    dt=dt,
                    x0=x0,
                    sol=None,
                    residual=r0,
                    elapsed_time=0.0,
                )

            solver_kwargs = self._least_squares_kwargs(cache, x0, least_squares_settings)

            start_time = time.perf_counter()
            sol = least_squares(**solver_kwargs)
            elapsed_time = time.perf_counter() - start_time

            # Push the candidate solution into the network and recompute all
            # derived states one final time at t_new.  The residual checked below
            # is the actual residual used for timestep acceptance.
            cache.assign_iteration_values(sol.x)
            self.network.time.value = t_new
            self.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance,
                cache=cache,
            )
            final_residual = cache.collect_residuals(dt)

            self._check_residual_vector(
                residual=final_residual,
                success=bool(sol.success),
                message=str(sol.message),
                settings=least_squares_settings,
                cache=cache,
                t_new=t_new,
                dt=dt,
            )
            # Evaluate once more after acceptance so exported algebraic outputs
            # reflect the accepted continuous state for this timestep.
            self.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance,
                cache=cache,
            )

            # Only after a timestep passes the residual check do we advance
            # transient history.  During the nonlinear solve,
            # ``state.previous`` remained the last accepted value while SciPy
            # changed ``state.value``.
            cache.store_previous_values()

            return StepDiagnostics(
                time=t_new,
                dt=dt,
                x0=x0,
                sol=sol,
                residual=final_residual,
                elapsed_time=elapsed_time,
            )

        except Exception:
            cache.restore_mutable_states(accepted_snapshot)
            self.network.time.value = accepted_time
            raise
        finally:
            self._active_cache = None

    def _least_squares_kwargs(
        self,
        cache,
        x0: np.ndarray,
        settings: LeastSquaresSettings,
    ) -> dict[str, Any]:
        """Build SciPy keyword arguments using bounds already stored on States."""
        kwargs: dict[str, Any] = {
            "fun": self.residual,
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
                raise SolverSetupError("solver_method='lm' does not support bounded States.")
            return kwargs

        # This is the same bound pattern used by the steady-state solver.  There
        # is no transient-specific bound system: the new-time unknowns simply use
        # their State.lower_bound / State.upper_bound / State.keep_feasible data.
        kwargs["bounds"] = Bounds(
            cache.lower_bound_array(),
            cache.upper_bound_array(),
            cache.keep_feasible_array(),
        )
        return kwargs

    def _check_residual_vector(
        self,
        *,
        residual: np.ndarray,
        success: bool,
        message: str,
        settings: LeastSquaresSettings,
        cache,
        t_new: float,
        dt: float,
    ) -> None:
        """Raise if the accepted residual is too large.

        SciPy is still run with strict nonlinear tolerances.  By default, transient
        solves disable SciPy ``gtol`` so gradient-based termination cannot stop
        early on a small-but-unaccepted integration residual.  After SciPy stops,
        timestep acceptance is based on the recomputed residual.
        The timestep is accepted only when the recomputed residual satisfies
        the configured per-step residual tolerance.
        """
        final_residual = np.array(residual, dtype=float)
        max_residual = np.max(np.abs(final_residual)) if len(final_residual) else 0.0

        if max_residual <= settings.rtol:
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
            "Transient timestep failed or converged to unacceptable residuals.",
            "The outer transient loop may retry this step with a smaller dt.",
            f"time = {t_new:.9g}",
            f"dt = {dt:.9g}",
            f"success = {success}",
            f"message = {message}",
            f"max |accepted residual| = {max_residual:.3e}",
            f"per-step residual tolerance = {settings.rtol:.3e}",
        ]

        if worst_rows:
            lines.extend(["", "Largest residuals:"])
            lines.extend(
                f"  - {label}: {value:.6e}"
                for label, value in worst_rows
            )

        raise TransientStepError("\n".join(lines))
