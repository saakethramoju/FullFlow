"""Executable transient timestep operations.

Executes fixed-step implicit backward Euler using scipy.optimize.least_squares.
The public component/solver API and nonlinear algorithm are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import inspect
import time

import numpy as np
from scipy.optimize import Bounds, least_squares

from fullflow.Solvers.steady_state.settings import LeastSquaresSettings, StateEvaluationSettings
from fullflow.Exceptions import SensorDataStop, SolverSetupError, TransientStepError


# Older SciPy releases lack the least_squares iteration callback. In those
# environments FullFlow retains the original SciPy convergence behavior.
_HAS_LEAST_SQUARES_CALLBACK = "callback" in inspect.signature(least_squares).parameters


@dataclass(slots=True)
class StepDiagnostics:
    """Diagnostics for one accepted implicit timestep."""

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

    def __init__(self, network, cache_getter: Callable[[], Any], evaluator) -> None:
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
        self._invalid_residual_directional_gain = 10.0
        self._reset_prediction_history()

    def _reset_prediction_history(self) -> None:
        """Forget prior accepted steps at the beginning of a new solve."""
        self._previous_step_x: np.ndarray | None = None
        self._accepted_step_x: np.ndarray | None = None
        self._accepted_step_time: float | None = None
        self._accepted_step_dt: float | None = None
        self._accepted_step_ids: tuple[int, ...] | None = None
        self._accepted_step_version: int | None = None
        self._accepted_residual_count: int | None = None

    def _record_accepted_step(
        self, cache, previous_x: np.ndarray, t_new: float, dt: float, residual_count: int,
    ) -> None:
        """Store history only after a successful step, never after a rejected trial."""
        self._previous_step_x = np.array(previous_x, dtype=float, copy=True)
        self._accepted_step_x = cache.iteration_value_array().copy()
        self._accepted_step_time = float(t_new)
        self._accepted_step_dt = float(dt)
        self._accepted_step_ids = tuple(id(state) for state in cache.iteration_variables)
        self._accepted_step_version = getattr(cache, "version", None)
        self._accepted_residual_count = residual_count

    def _predict_initial_guess(
        self, cache, x0: np.ndarray, old_time: float, dt: float, residual_count: int,
    ) -> np.ndarray | None:
        """Extrapolate the generic unknown vector from two accepted time levels.

        Return None whenever history, unknown ordering, time or State bounds
        have changed. No component or State type needs special handling.
        """
        prior = self._previous_step_x
        last = self._accepted_step_x
        old_dt = self._accepted_step_dt
        last_time = self._accepted_step_time
        if prior is None or last is None or old_dt is None or last_time is None:
            return None
        if old_dt <= 0.0 or dt <= 0.0 or dt > 2.0 * old_dt:
            return None
        if abs(last_time - old_time) > 1e-12 * max(1.0, abs(old_time)):
            return None
        if self._accepted_step_version != getattr(cache, "version", None):
            return None
        if self._accepted_step_ids != tuple(id(state) for state in cache.iteration_variables):
            return None
        if self._accepted_residual_count != residual_count:
            return None
        if prior.shape != x0.shape or last.shape != x0.shape:
            return None
        if not np.array_equal(last, x0):
            return None

        guess = x0 + (dt / old_dt) * (x0 - prior)
        if not np.all(np.isfinite(guess)) or np.array_equal(guess, x0):
            return None
        if np.any(guess < cache.lower_bound_array()) or np.any(guess > cache.upper_bound_array()):
            return None
        return guess

    def _restore_trial_start(
        self, cache, snapshot, x0: np.ndarray, t_new: float, dt: float,
    ) -> np.ndarray:
        """Undo speculative evaluations before retrying from the accepted state."""
        cache.clear_sequence_command_pending()
        cache.restore_mutable_states(snapshot)
        self.network.time.value = t_new
        if hasattr(cache, "configure_sensor_balances"):
            cache.configure_sensor_balances(t_new, stop_on_missing=True)
        cache.set_transient_context(dt=dt)
        cache.apply_sequence_commands(t_new)
        return self.residual(x0)

    def _solve_with_early_acceptance(self, cache, x0: np.ndarray, settings):
        """Stop TRF/dogbox after a complete accepted iterate reaches rtol.

        SciPy calls this callback only after a nonlinear iteration, not while
        calculating finite-difference Jacobian columns. The original final
        network/residual verification remains mandatory.
        """
        kwargs = self._least_squares_kwargs(cache, x0, settings)
        early_stop = [False]
        if _HAS_LEAST_SQUARES_CALLBACK and settings.solver_method != "lm":
            def accepted_iteration(intermediate_result):
                values = getattr(intermediate_result, "fun", None)
                if values is None:
                    return
                values = np.asarray(values, dtype=float)
                if values.size and (not np.all(np.isfinite(values))):
                    return
                if not values.size or float(np.max(np.abs(values))) <= settings.rtol:
                    early_stop[0] = True
                    raise StopIteration

            kwargs["callback"] = accepted_iteration

        return least_squares(**kwargs), early_stop[0]

    @staticmethod
    def _mark_early_acceptance(sol, stopped_early: bool) -> None:
        """Report an accepted callback termination accurately in diagnostics."""
        if stopped_early and getattr(sol, "status", None) == -2:
            sol.success = True
            sol.message = "FullFlow accepted a converged residual at a SciPy iteration callback."


    def residual(self, x: np.ndarray) -> np.ndarray:
        """Compute current trial residual: dynamics, algebraic, then balances."""
        cache = self._active_cache or self._cache_getter()
        # A single owned snapshot serves both debug and last-valid recovery;
        # the incoming SciPy vector may be reused or mutated by its caller.
        trial_snapshot = np.array(x, dtype=float)
        self._last_debug_x = trial_snapshot
        self._last_debug_residual = None
        self._last_debug_error = None

        cache.assign_iteration_values(x)
        self.network.time.value = self._active_time
        cache.set_transient_context(dt=self._active_dt)
        cache.apply_sequence_commands(self._active_time)

        try:
            self.evaluator.run(
                max_passes=self._active_state_settings.max_passes,
                tolerance=self._active_state_settings.tolerance,
                cache=cache,
            )
            residual = cache.collect_residuals(self._active_dt)
            self._last_debug_residual = residual
            self._last_valid_x = trial_snapshot
            # Preserve an independent copy for invalid-trial recovery.
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
        """Preserve the existing directional penalty for invalid trial states."""
        if self._last_valid_residual is None:
            return None

        last_residual = np.array(self._last_valid_residual, dtype=float)
        count = len(last_residual)
        if count == 0:
            return np.array([], dtype=float)

        if self._last_valid_x is None or len(x) == 0:
            direction = np.sign(last_residual)
            direction[direction == 0.0] = 1.0
            distance = 1.0
        else:
            dx = np.array(x, dtype=float) - self._last_valid_x
            scale = np.maximum(np.abs(self._last_valid_x), 1.0)
            normalized_dx = dx / scale
            distance = float(np.linalg.norm(normalized_dx))
            if distance == 0.0 or not np.isfinite(distance):
                direction = np.sign(last_residual)
                direction[direction == 0.0] = 1.0
                distance = 1.0
            else:
                direction = np.resize(normalized_dx / distance, count)

        last_norm = float(np.linalg.norm(last_residual))
        residual_rms = float(np.sqrt(np.mean(last_residual**2)))
        residual_scale = max(residual_rms, 1.0)
        target_norm = max(
            last_norm * (1.0 + self._invalid_residual_directional_gain * distance),
            residual_scale * np.sqrt(count),
        )
        residual = last_residual + (
            self._invalid_residual_directional_gain * residual_scale * distance * direction
        )
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm == 0.0 or not np.isfinite(residual_norm):
            residual = residual_scale * direction
            residual_norm = float(np.linalg.norm(residual))
        if residual_norm < target_norm:
            residual *= target_norm / max(residual_norm, np.finfo(float).tiny)
        return residual

    def initialize(self, state_settings: StateEvaluationSettings) -> None:
        """Evaluate the initial network and store accepted transient history."""
        state_settings.validate()
        self._reset_prediction_history()
        cache = self._cache_getter()
        cache.run_pre_evaluation()
        cache = self._cache_getter()
        self._active_cache = cache
        try:
            cache.set_transient_context(dt=0.0)
            cache.apply_sequence_commands(float(self.network.time.value))
            self.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance,
                cache=cache,
            )
            cache.store_previous_values()
            cache.commit_sequence_commands()
        finally:
            cache.clear_sequence_command_pending()
            self._active_cache = None



    def run_once(
        self,
        *,
        t_new: float,
        dt: float,
        least_squares_settings: LeastSquaresSettings,
        state_settings: StateEvaluationSettings,
    ) -> StepDiagnostics:
        """Solve and accept one implicit backward-Euler timestep."""
        least_squares_settings.validate()
        state_settings.validate()
        self._active_time = float(t_new)
        self._active_dt = float(dt)
        self._active_state_settings = state_settings

        cache = self._cache_getter()
        accepted_snapshot = cache.snapshot_mutable_states()
        accepted_time = float(self.network.time.value)
        self.network.time.value = t_new
        if hasattr(cache, "configure_sensor_balances"):
            cache.configure_sensor_balances(t_new, stop_on_missing=True)
        cache.set_transient_context(dt=dt)
        cache.apply_sequence_commands(t_new)
        self._active_cache = cache
        x0 = cache.iteration_value_array()
        self._last_valid_x = None
        self._last_valid_residual = None

        try:
            predictor_start = time.perf_counter()
            self.network.time.value = t_new
            r0 = self.residual(x0)
            predictor_elapsed = time.perf_counter() - predictor_start
            cache.validate_residual_count(r0)

            max_r0 = float(np.max(np.abs(r0))) if len(r0) else 0.0
            if max_r0 <= least_squares_settings.rtol:
                cache.apply_sequence_commands(t_new)
                self.evaluator.run(
                    max_passes=state_settings.max_passes,
                    tolerance=state_settings.tolerance,
                    cache=cache,
                )
                cache.store_previous_values()
                cache.commit_sequence_commands()
                self._record_accepted_step(cache, x0, t_new, dt, len(r0))
                return StepDiagnostics(
                    time=t_new, dt=dt, x0=x0, sol=None,
                    residual=r0, elapsed_time=predictor_elapsed,
                )

            if len(x0) == 0:
                self._check_residual_vector(
                    residual=r0,
                    success=True,
                    message="No nonlinear variables; static transient evaluation.",
                    settings=least_squares_settings,
                    cache=cache,
                    t_new=t_new,
                    dt=dt,
                )
                cache.apply_sequence_commands(t_new)
                self.evaluator.run(
                    max_passes=state_settings.max_passes,
                    tolerance=state_settings.tolerance,
                    cache=cache,
                )
                cache.store_previous_values()
                cache.commit_sequence_commands()
                self._record_accepted_step(cache, x0, t_new, dt, len(r0))
                return StepDiagnostics(
                    time=t_new, dt=dt, x0=x0, sol=None,
                    residual=r0, elapsed_time=predictor_elapsed,
                )

            # An optional, generic initial guess. Test it against the same
            # backward-Euler residual before giving it to SciPy. One extra
            # residual evaluation can save many finite-difference evaluations.
            guess = self._predict_initial_guess(cache, x0, accepted_time, dt, len(r0))
            use_prediction = False
            if guess is not None:
                try:
                    predicted_residual = self.residual(guess)
                    max_predicted = (
                        float(np.max(np.abs(predicted_residual)))
                        if len(predicted_residual) else 0.0
                    )
                    use_prediction = (
                        self._last_debug_error is None
                        and len(predicted_residual) == len(r0)
                        and np.all(np.isfinite(predicted_residual))
                        and (max_predicted <= least_squares_settings.rtol
                             or max_predicted < 0.95 * max_r0)
                    )
                except SensorDataStop:
                    raise
                except Exception:
                    use_prediction = False

                if not use_prediction:
                    # Restore all mutable States, not just SciPy unknowns.
                    r0 = self._restore_trial_start(
                        cache, accepted_snapshot, x0, t_new, dt,
                    )
                    cache.validate_residual_count(r0)

            initial_guess = guess if use_prediction else x0
            start_time = time.perf_counter()

            # A successful predicted residual may already satisfy the BE
            # equations. Verify it once more before using the fast path.
            if use_prediction and max_predicted <= least_squares_settings.rtol:
                cache.assign_iteration_values(guess)
                self.network.time.value = t_new
                cache.apply_sequence_commands(t_new)
                self.evaluator.run(
                    max_passes=state_settings.max_passes,
                    tolerance=state_settings.tolerance,
                    cache=cache,
                )
                predicted_final = cache.collect_residuals(dt)
                if (np.all(np.isfinite(predicted_final)) and
                        len(predicted_final) == len(r0) and
                        (not len(predicted_final) or
                         float(np.max(np.abs(predicted_final))) <= least_squares_settings.rtol)):
                    cache.apply_sequence_commands(t_new)
                    self.evaluator.run(
                        max_passes=state_settings.max_passes,
                        tolerance=state_settings.tolerance,
                        cache=cache,
                    )
                    cache.store_previous_values()
                    cache.commit_sequence_commands()
                    self._record_accepted_step(cache, x0, t_new, dt, len(predicted_final))
                    return StepDiagnostics(
                        time=t_new, dt=dt, x0=guess, sol=None,
                        residual=predicted_final,
                        elapsed_time=predictor_elapsed + time.perf_counter() - start_time,
                    )
                # A non-repeatable component invalidated the shortcut.
                # Start again from the last accepted state.
                use_prediction = False
                initial_guess = x0
                self._restore_trial_start(cache, accepted_snapshot, x0, t_new, dt)

            # The existing SciPy solver and equations are unchanged. Predictive
            # starts get one automatic retry from the original accepted guess
            # if the candidate fails its *actual* final residual check.
            try:
                sol, stopped_early = self._solve_with_early_acceptance(
                    cache, initial_guess, least_squares_settings,
                )
                cache.assign_iteration_values(sol.x)
                self.network.time.value = t_new
                cache.apply_sequence_commands(t_new)
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
            except SensorDataStop:
                raise
            except Exception:
                if not use_prediction:
                    raise
                self._restore_trial_start(cache, accepted_snapshot, x0, t_new, dt)
                initial_guess = x0
                sol, stopped_early = self._solve_with_early_acceptance(
                    cache, x0, least_squares_settings,
                )
                cache.assign_iteration_values(sol.x)
                self.network.time.value = t_new
                cache.apply_sequence_commands(t_new)
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

            self._mark_early_acceptance(sol, stopped_early)
            elapsed_time = time.perf_counter() - start_time
            cache.apply_sequence_commands(t_new)
            self.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance,
                cache=cache,
            )
            cache.store_previous_values()
            cache.commit_sequence_commands()
            self._record_accepted_step(cache, x0, t_new, dt, len(final_residual))
            return StepDiagnostics(
                time=t_new, dt=dt, x0=initial_guess, sol=sol,
                residual=final_residual, elapsed_time=elapsed_time,
            )

        except Exception:
            cache.clear_sequence_command_pending()
            cache.restore_mutable_states(accepted_snapshot)
            self.network.time.value = accepted_time
            raise
        finally:
            cache.clear_sequence_command_pending()
            self._active_cache = None

    def _least_squares_kwargs(
        self, cache, x0: np.ndarray, settings: LeastSquaresSettings,
    ) -> dict[str, Any]:
        """Construct the original scipy.optimize.least_squares arguments."""
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
        kwargs["bounds"] = Bounds(
            cache.lower_bound_array(),
            cache.upper_bound_array(),
            cache.keep_feasible_array(),
        )
        return kwargs

    def _check_residual_vector(
        self, *, residual: np.ndarray, success: bool, message: str,
        settings: LeastSquaresSettings, cache, t_new: float, dt: float,
    ) -> None:
        """Preserve the existing residual tolerance and failure diagnostics."""
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
                f"residual[{i}]" for i in range(len(residual_labels), len(final_residual))
            )
        worst_rows = sorted(
            zip(residual_labels, final_residual),
            key=lambda pair: abs(pair[1]), reverse=True,
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
            lines.extend(f"  - {label}: {value:.6e}" for label, value in worst_rows)
        raise TransientStepError("\n".join(lines))
