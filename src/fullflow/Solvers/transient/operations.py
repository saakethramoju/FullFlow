"""Transient timestep evaluation and nonlinear solution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
import inspect
import time

import numpy as np
from scipy.optimize import Bounds, least_squares

from fullflow.Solvers.steady_state.settings import LeastSquaresSettings, StateEvaluationSettings
from fullflow.Exceptions import SensorDataStop, SolverSetupError, TransientStepError
from .newton import ModifiedNewtonFailure, solve_modified_newton
from fullflow.Solvers.equation_scaling import EquationScaling


_HAS_LEAST_SQUARES_CALLBACK = "callback" in inspect.signature(least_squares).parameters


_ENABLE_MODIFIED_NEWTON_BROYDEN = True


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
    scaled_residual: np.ndarray | None = None

    @property
    def max_scaled_residual(self) -> float:
        """Maximum residual used for the solver's scaled convergence test."""
        values = self.scaled_residual if self.scaled_residual is not None else self.residual
        return float(np.max(np.abs(values))) if len(values) else 0.0

    @property
    def max_residual(self) -> float:
        """Return the maximum absolute step residual."""
        return float(np.max(np.abs(self.residual))) if len(self.residual) else 0.0

    @property
    def rms_residual(self) -> float:
        """Return the root-mean-square step residual."""
        return float(np.sqrt(np.mean(self.residual**2))) if len(self.residual) else 0.0


class TransientStepSolve:
    """Solve one implicit timestep with safeguarded Newton and SciPy fallback."""

    def __init__(self, network, cache_getter: Callable[[], Any], evaluator) -> None:
        """Initialize the helper and its solver state."""
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
        self._active_equation_scaling: EquationScaling | None = None
        self._reset_prediction_history()
        self._pending_trial = None
        self.work_statistics = {}
        self._reset_work_statistics()

    def _reset_work_statistics(self):
        self.work_statistics = dict.fromkeys((
            "implicit_step_calls", "residual_calls", "newton_attempts",
            "newton_jacobian_builds", "newton_rejections", "scipy_solves",
            "scipy_reported_jacobian_builds", "scale_jacobian_builds",
            "one_sided_jacobian_steps", "event_jacobian_seed_uses",
            "event_jacobian_seed_rejections",
        ), 0)

    def _count_newton_jacobian(self):
        self.work_statistics["newton_jacobian_builds"] += 1

    def _clear_newton_jacobian(self) -> None:
        """Discard any cached modified-Newton Jacobian."""
        self._newton_jacobian: np.ndarray | None = None
        self._newton_jacobian_scale: np.ndarray | None = None
        self._newton_jacobian_method: str | None = None
        self._newton_cache_id: int | None = None
        self._newton_control_values = ()
        self._newton_reuse_age = 0

    def _reset_prediction_history(self) -> None:
        """Forget prior accepted steps at the beginning of a new solve."""
        self._clear_newton_jacobian()
        self._scipy_refinement = None
        self._encountered_nonsmooth_control = False
        self._accepted_equation_scales: np.ndarray | None = None
        self._previous_step_x: np.ndarray | None = None
        self._accepted_step_x: np.ndarray | None = None
        self._accepted_step_time: float | None = None
        self._accepted_step_dt: float | None = None
        self._accepted_step_ids: tuple[int, ...] | None = None
        self._accepted_step_version: int | None = None
        self._accepted_residual_count: int | None = None
        self._accepted_control_values: tuple | None = None
        self._control_source_key = None
        self._control_sources = ()

    def _make_diagnostics(self, **kwargs) -> StepDiagnostics:
        result = StepDiagnostics(**kwargs)
        scaling = self._active_equation_scaling
        if scaling is not None:
            result.scaled_residual = scaling.residual(result.residual).copy()
        return result

    def invalidate_equation_scales(self) -> None:
        """Invalidate linearization and equation scales after an operating-mode event."""
        self._clear_newton_jacobian()
        self._accepted_equation_scales = None
        self._previous_step_x = None

    def _residual_norm(self, residual: np.ndarray) -> float:
        scaling = self._active_equation_scaling
        values = scaling.residual(residual) if scaling is not None else residual
        return float(np.max(np.abs(values))) if len(values) else 0.0

    def _scaling_for_step(self, cache, x0, accepted_time, dt, residual_count):
        scales = self._accepted_equation_scales
        if not (
            scales is not None
            and self._accepted_step_time is not None
            and abs(self._accepted_step_time - accepted_time) <= 1e-12 * max(1.0, abs(accepted_time))
            and self._accepted_step_dt is not None
            and abs(self._accepted_step_dt - dt) <= 1e-12 * max(1.0, abs(dt))
            and self._accepted_step_version == getattr(cache, "version", None)
            and self._accepted_step_ids == tuple(id(state) for state in cache.iteration_variables)
            and self._accepted_residual_count == residual_count
            and self._accepted_step_x is not None
            and np.array_equal(self._accepted_step_x, x0)
            and not self._controls_changed(self._accepted_control_values or (), self._control_values(cache))
        ):
            scales = None
            self._clear_newton_jacobian()
        return EquationScaling.transient(cache, residual_count, scales)

    def _finish_trial(self, cache, previous_x, t_new, dt, residual_count, *, commit=True, **jacobian_options):
        """Stage a solved candidate; only accepted trials advance State history."""
        if commit:
            cache.store_previous_values()
            cache.commit_sequence_commands()
            self._record_accepted_step(cache, previous_x, t_new, dt, residual_count, **jacobian_options)
        else:
            self._pending_trial = (
                cache, previous_x, t_new, dt, residual_count, jacobian_options,
                self._active_equation_scaling,
            )

    def accept_trial(self):
        """Commit a previously solved candidate without another nonlinear solve."""
        if self._pending_trial is None:
            raise RuntimeError("No successfully solved transient trial is pending.")
        cache, previous_x, t_new, dt, residual_count, options, scaling = self._pending_trial
        self._pending_trial = None
        cache.store_previous_values()
        cache.commit_sequence_commands()
        self._active_equation_scaling = scaling
        try:
            self._record_accepted_step(cache, previous_x, t_new, dt, residual_count, **options)
        finally:
            self._active_equation_scaling = None

    def event_jacobian_seed(self):
        """Return the verified finite-difference model for a pending event trial."""
        pending = self._pending_trial
        if pending is None:
            return None
        cache, _, _, dt, residual_count, options, scaling = pending
        jac = options.get("jacobian")
        scale = options.get("jacobian_scale")
        if (jac is None or scale is None or scaling is None
                or scaling.scales is None):
            return None
        return (
            np.array(jac, copy=True), np.array(scale, copy=True),
            np.array(scaling.scales, copy=True), dt,
            getattr(cache, "version", None),
            tuple(id(state) for state in cache.iteration_variables),
            residual_count, options.get("jacobian_method"),
            cache.iteration_labels,
            frozenset(getattr(cache, "_active_sensor_balance_owner_ids", ())),
            frozenset(getattr(cache, "_direct_sensor_balance_owner_ids", ())),
        )

    def discard_trial(self):
        """Invalidate trial and predictor history after restoring the accepted point."""
        self._pending_trial = None
        self._reset_prediction_history()

    def _record_accepted_step(
        self, cache, previous_x: np.ndarray, t_new: float, dt: float, residual_count: int,
        *, jacobian=None, jacobian_scale=None, jacobian_method=None,
        jacobian_reused=False, jacobian_rebuilt=False,
    ) -> None:
        """Store history after an accepted step."""
        reuse_age = self._newton_reuse_age + 1 if jacobian_reused and not jacobian_rebuilt else 0
        self._clear_newton_jacobian()
        self._previous_step_x = np.array(previous_x, dtype=float, copy=True)
        self._accepted_step_x = cache.iteration_value_array().copy()
        self._accepted_step_time = float(t_new)
        self._accepted_step_dt = float(dt)
        self._accepted_step_ids = tuple(id(state) for state in cache.iteration_variables)
        self._accepted_step_version = getattr(cache, "version", None)
        self._accepted_residual_count = residual_count
        self._accepted_control_values = self._control_values(cache)
        scaling = self._active_equation_scaling
        self._accepted_equation_scales = (
            scaling.scales.copy() if scaling is not None and scaling.scales is not None
            else None
        )
        if jacobian is not None and jacobian_scale is not None:
            matrix = np.asarray(jacobian, dtype=float)
            scale = np.asarray(jacobian_scale, dtype=float)
            if (matrix.shape == (residual_count, len(self._accepted_step_x))
                    and scale.shape == self._accepted_step_x.shape
                    and np.all(np.isfinite(matrix))
                    and np.all(np.isfinite(scale)) and np.all(scale > 0.0)):
                self._newton_jacobian = matrix.copy()
                self._newton_jacobian_scale = scale.copy()
                self._newton_jacobian_method = jacobian_method
                self._newton_cache_id = id(cache)
                self._newton_control_values = self._accepted_control_values
                self._newton_reuse_age = reuse_age

    def _control_values(self, cache) -> tuple:
        """Return the current transient control values."""
        components = tuple(getattr(cache, "component_list", ()))
        component_ids = tuple(id(component) for component in components)
        cache_key = (id(cache), component_ids)
        if self._control_source_key != cache_key:
            sources = []
            for component in components:
                if "_has_primary_sequence" in getattr(component, "__dict__", ()) or getattr(component, "_has_primary_sequence", False):
                    sources.append((component, True))
                if getattr(type(component), "command_entries", None) is not None:
                    sources.append((component, False))
            self._control_sources = tuple(sources)
            self._control_source_key = cache_key

        values = []
        for component, primary in self._control_sources:
            if primary:
                if not getattr(component, "_has_primary_sequence", False):
                    continue
                target = getattr(component, "target", None)
                if target is not None:
                    try:
                        values.append((
                            id(target),
                            'sampled' if getattr(component, '_sample_enabled', False) else True,
                            float(target.value),
                        ))
                    except (TypeError, ValueError, AttributeError):
                        pass
            else:
                for entry in component.command_entries:
                    try:
                        values.append((id(entry.target), bool(entry.is_active), float(entry.target.value)))
                    except (TypeError, ValueError, AttributeError):
                        pass
        return tuple(values)

    @staticmethod
    def _controls_changed(previous: tuple, current: tuple) -> bool:
        """Return whether transient controls changed."""
        if len(previous) != len(current):
            return True
        for old, new in zip(previous, current):
            if old[:2] != new[:2]:
                return True
            a, b = old[2], new[2]
            if not np.isfinite(a) or not np.isfinite(b):
                return True
            if old[1] == 'sampled':
                threshold = 0.25 * max(1.0, abs(a), abs(b))
            else:
                threshold = 0.25 * max(abs(a), abs(b), 1.0e-12)
            if abs(a - b) > threshold:
                return True
        return False

    def _initial_newton_jacobian(
        self, cache, x0: np.ndarray, old_time: float, dt: float,
        residual_count: int, jacobian_method: str,
        *, probe_x=None, probe_residual=None, baseline_residual=None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Build or reuse the initial Newton Jacobian."""
        if (self._newton_jacobian is None or self._newton_jacobian_scale is None
                or self._active_equation_scaling is None
                or self._active_equation_scaling.scales is None):
            return None, None
        valid = (
            self._newton_reuse_age < 8
            and self._newton_cache_id == id(cache)
            and self._newton_jacobian_method == jacobian_method
            and self._accepted_step_time is not None
            and abs(self._accepted_step_time - old_time) <= 1.0e-12 * max(1.0, abs(old_time))
            and self._accepted_step_dt is not None
            and abs(self._accepted_step_dt - dt) <= 1.0e-12 * max(1.0, abs(dt))
            and self._accepted_step_version == getattr(cache, "version", None)
            and self._accepted_step_ids == tuple(id(state) for state in cache.iteration_variables)
            and self._accepted_residual_count == residual_count
            and self._accepted_step_x is not None
            and np.array_equal(self._accepted_step_x, x0)
            and not self._controls_changed(
                self._newton_control_values, self._control_values(cache),
            )
        )
        if not valid:
            self._clear_newton_jacobian()
            return None, None

        probe_error = None
        if probe_x is not None and probe_residual is not None and baseline_residual is not None:
            delta = (np.asarray(probe_x, dtype=float) - x0) / self._newton_jacobian_scale
            if np.all(np.isfinite(delta)) and float(np.max(np.abs(delta))) > 1.0e-9:
                actual = np.asarray(probe_residual, dtype=float) - baseline_residual
                if self._active_equation_scaling is not None:
                    actual = self._active_equation_scaling.residual(actual)
                expected = self._newton_jacobian @ delta
                divisor = max(float(np.linalg.norm(actual)), float(np.linalg.norm(expected)))
                if divisor > 0.0 and np.isfinite(divisor):
                    probe_error = float(np.linalg.norm(actual - expected)) / divisor
                else:
                    probe_error = float("inf")

        if (probe_error is not None and (not np.isfinite(probe_error) or probe_error > 0.65)) or (
            self._newton_reuse_age >= 4 and (probe_error is None or probe_error > 0.35)
        ):
            self._clear_newton_jacobian()
            return None, None
        return self._newton_jacobian, self._newton_jacobian_scale

    def _predict_initial_guess(
        self, cache, x0: np.ndarray, old_time: float, dt: float, residual_count: int,
    ) -> np.ndarray | None:
        """Extrapolate the generic unknown vector from two accepted time levels."""
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
        if (self._accepted_control_values is None or
                self._controls_changed(self._accepted_control_values, self._control_values(cache))):
            return None
        if prior.shape != x0.shape or last.shape != x0.shape:
            return None
        if not np.array_equal(last, x0):
            return None

        guess = x0 + (dt / old_dt) * (x0 - prior)
        if not np.all(np.isfinite(guess)) or np.array_equal(guess, x0):
            return None
        normalized_change = np.abs(guess - x0) / np.maximum(
            np.maximum(np.abs(x0), np.abs(prior)), 1.0,
        )
        if float(np.max(normalized_change)) > 2.0:
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
        """Stop TRF/dogbox after a complete accepted iterate reaches rtol."""
        scaling = self._active_equation_scaling
        if scaling is not None and scaling.scales is None:
            self.work_statistics["scale_jacobian_builds"] += 1
            bounds = (cache.lower_bound_array(), cache.upper_bound_array())
            scaling.initialize(self.residual, x0, settings.jacobian_method, bounds=bounds)
        kwargs = self._least_squares_kwargs(cache, x0, settings)
        refinement_key = (id(cache), settings.solver_method, settings.jacobian_method)

        def refine_kwargs(diff_step):
            kwargs.update(
                jac="2-point", diff_step=diff_step, max_nfev=100,
                ftol=None, xtol=None, gtol=10.0 * np.finfo(float).eps,
            )

        if (self._scipy_refinement is not None
                and self._scipy_refinement[:3] == refinement_key):
            refine_kwargs(self._scipy_refinement[3])
        early_stop = [False]
        if _HAS_LEAST_SQUARES_CALLBACK and settings.solver_method != "lm":
            def accepted_iteration(intermediate_result):
                """Record an accepted intermediate Newton iteration."""
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

        self.work_statistics["scipy_solves"] += 1
        sol = least_squares(**kwargs)
        self.work_statistics["scipy_reported_jacobian_builds"] += int(sol.njev or 0)
        nfev, njev = sol.nfev, sol.njev
        for diff_step in (np.sqrt(np.finfo(float).eps), np.finfo(float).eps ** 0.75):
            if (settings.solver_method == "lm" or not np.all(np.isfinite(sol.fun))
                    or np.max(np.abs(sol.fun), initial=0.0) <= settings.rtol):
                break
            # Near zero pressure drop, central differences can span both flow
            # branches. Large absolute pressures can also trigger xtol while
            # the conservation equations are still far from converged.
            # Refine the current iterate with a smaller, one-sided Jacobian;
            # acceptance still requires the original residual tolerance.
            kwargs["x0"] = sol.x
            refine_kwargs(diff_step)
            refined = least_squares(**kwargs)
            nfev += refined.nfev
            if njev is not None and refined.njev is not None:
                njev += refined.njev
            if (np.all(np.isfinite(refined.fun))
                    and np.max(np.abs(refined.fun), initial=0.0)
                    < np.max(np.abs(sol.fun), initial=0.0)):
                sol = refined
                if np.max(np.abs(sol.fun), initial=0.0) <= settings.rtol:
                    # Keep a successful refinement for subsequent steps in
                    # this runtime rather than repeatedly exhausting a coarse
                    # Jacobian near the same zero-flow operating mode.
                    self._scipy_refinement = (*refinement_key, diff_step)
        sol.nfev, sol.njev = nfev, njev
        return sol, early_stop[0]

    @staticmethod
    def _may_try_modified_newton(cache, settings, x0, r0) -> bool:
        """Determine whether the Newton solver can handle this system."""
        if not _ENABLE_MODIFIED_NEWTON_BROYDEN or settings.solver_method != "trf":
            return False
        if not len(x0) or len(r0) < len(x0):
            return False
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(r0)):
            return False
        try:
            if (np.any(np.isfinite(cache.lower_bound_array()))
                    or np.any(np.isfinite(cache.upper_bound_array()))
                    or np.any(cache.keep_feasible_array())):
                return False
        except Exception:
            return False
        return True

    def _try_modified_newton(self, cache, x0, r0, settings, jacobian=None, jacobian_scale=None):
        """Solve and verify a timestep using Newton/Broyden."""
        self.work_statistics["newton_attempts"] += 1
        candidate = solve_modified_newton(
            self.residual, x0, r0,
            rtol=settings.rtol,
            jacobian_method=settings.jacobian_method,
            initial_jacobian=jacobian,
            initial_scale=jacobian_scale,
            equation_scaling=self._active_equation_scaling,
            on_jacobian_build=self._count_newton_jacobian,
        )


        cache.assign_iteration_values(candidate.x)
        self.network.time.value = self._active_time
        cache.apply_sequence_commands(self._active_time)
        self.evaluator.run(
            max_passes=self._active_state_settings.max_passes,
            tolerance=self._active_state_settings.tolerance,
            cache=cache,
        )
        candidate_final = cache.collect_residuals(self._active_dt)
        self._check_residual_vector(
            residual=candidate_final, success=True,
            message=str(candidate.message), settings=settings,
            cache=cache, t_new=self._active_time, dt=self._active_dt,
        )
        if (len(candidate_final) != len(r0)
                or not np.all(np.isfinite(candidate_final))):
            raise ModifiedNewtonFailure("Final Newton residual shape/values changed.")
        candidate.fun = np.array(candidate_final, dtype=float)
        candidate.cost = 0.5 * float(np.dot(candidate_final, candidate_final))
        candidate.optimality = float(np.max(np.abs(candidate_final))) if len(candidate_final) else 0.0
        return candidate, candidate_final

    @staticmethod
    def _mark_early_acceptance(sol, stopped_early: bool) -> None:
        """Report an accepted callback termination accurately in diagnostics."""
        if stopped_early and getattr(sol, "status", None) == -2:
            sol.success = True
            sol.message = "FullFlow accepted a converged residual at a SciPy iteration callback."


    def residual(self, x: np.ndarray) -> np.ndarray:
        """Compute current trial residual: dynamics, algebraic, then balances."""
        self.work_statistics["residual_calls"] += 1
        cache = self._active_cache or self._cache_getter()


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
        self._reset_work_statistics()
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
        self._active_equation_scaling = None


    def run_once(
        self,
        *,
        t_new: float,
        dt: float,
        least_squares_settings: LeastSquaresSettings,
        state_settings: StateEvaluationSettings,
        commit: bool = True,
        trial_guess: np.ndarray | None = None,
        event_jacobian_seed=None,
    ) -> StepDiagnostics:
        """Solve an implicit timestep; optionally defer acceptance for event checks."""
        self.work_statistics["implicit_step_calls"] += 1
        least_squares_settings.validate()
        state_settings.validate()
        self._active_time = float(t_new)
        self._active_dt = float(dt)
        self._active_state_settings = state_settings

        self._pending_trial = None
        cache = self._cache_getter()
        accepted_snapshot = cache.snapshot_mutable_states()
        accepted_time = float(self.network.time.value)
        try:
            self.network.time.value = t_new
            if hasattr(cache, "configure_sensor_balances"):
                cache.configure_sensor_balances(t_new, stop_on_missing=True)
            cache.set_transient_context(dt=dt)
            cache.apply_sequence_commands(t_new)
            self._active_cache = cache
            x0 = cache.iteration_value_array()
            self._last_valid_x = None
            self._last_valid_residual = None

            predictor_start = time.perf_counter()
            self.network.time.value = t_new
            r0 = self.residual(x0)
            predictor_elapsed = time.perf_counter() - predictor_start
            cache.validate_residual_count(r0)

            self._active_equation_scaling = self._scaling_for_step(
                cache, x0, accepted_time, dt, len(r0),
            )
            if event_jacobian_seed is not None:
                (jac, varscale, rowscale, old_dt, version, ids, count, method,
                 labels, active_sensors, direct_sensors) = event_jacobian_seed
                if (least_squares_settings.solver_method == "trf"
                        and method == least_squares_settings.jacobian_method
                        and old_dt > 0.0 and 0.5 <= dt / old_dt <= 2.0
                        and version == getattr(cache, "version", None)
                        and ids == tuple(id(state) for state in cache.iteration_variables)
                        and labels == cache.iteration_labels
                        and active_sensors == frozenset(getattr(cache, "_active_sensor_balance_owner_ids", ()))
                        and direct_sensors == frozenset(getattr(cache, "_direct_sensor_balance_owner_ids", ()))
                        and count == len(r0)
                        and rowscale.shape == r0.shape
                        and jac.shape == (len(r0), len(x0))
                        and varscale.shape == x0.shape
                        and np.all(np.isfinite(jac))
                        and np.all(np.isfinite(varscale)) and np.all(varscale > 0.0)
                        and np.all(np.isfinite(rowscale)) and np.all(rowscale > 0.0)):
                    self._active_equation_scaling.scales = rowscale.copy()
                else:
                    event_jacobian_seed = None
                    self.work_statistics["event_jacobian_seed_rejections"] += 1
            max_r0 = self._residual_norm(r0)
            if max_r0 <= least_squares_settings.rtol:
                self._finish_trial(cache, x0, t_new, dt, len(r0), commit=commit)
                return self._make_diagnostics(
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
                self._finish_trial(cache, x0, t_new, dt, len(r0), commit=commit)
                return self._make_diagnostics(
                    time=t_new, dt=dt, x0=x0, sol=None,
                    residual=r0, elapsed_time=predictor_elapsed,
                )


            guess = self._predict_initial_guess(cache, x0, accepted_time, dt, len(r0))
            if trial_guess is not None:
                candidate = np.asarray(trial_guess, dtype=float)
                if (candidate.shape == x0.shape and np.all(np.isfinite(candidate))
                        and np.all(candidate >= cache.lower_bound_array())
                        and np.all(candidate <= cache.upper_bound_array())):
                    guess = candidate
            use_prediction = False
            probe_x = None
            probe_residual = None
            if guess is not None:
                try:
                    predicted_residual = self.residual(guess)
                    max_predicted = self._residual_norm(predicted_residual)
                    prediction_valid = (
                        self._last_debug_error is None
                        and len(predicted_residual) == len(r0)
                        and np.all(np.isfinite(predicted_residual))
                    )
                    if prediction_valid:
                        probe_x = guess
                        probe_residual = predicted_residual.copy()
                    use_prediction = (
                        prediction_valid
                        and (max_predicted <= least_squares_settings.rtol
                             or max_predicted < 0.95 * max_r0)
                    )
                except SensorDataStop:
                    raise
                except Exception:
                    use_prediction = False

                if not use_prediction:

                    r0 = self._restore_trial_start(
                        cache, accepted_snapshot, x0, t_new, dt,
                    )
                    cache.validate_residual_count(r0)

            initial_guess = guess if use_prediction else x0
            start_time = time.perf_counter()


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
                         self._residual_norm(predicted_final) <= least_squares_settings.rtol)):
                    self._finish_trial(cache, x0, t_new, dt, len(predicted_final), commit=commit)
                    return self._make_diagnostics(
                        time=t_new, dt=dt, x0=guess, sol=None,
                        residual=predicted_final,
                        elapsed_time=predictor_elapsed + time.perf_counter() - start_time,
                    )


                use_prediction = False
                initial_guess = x0
                self._restore_trial_start(cache, accepted_snapshot, x0, t_new, dt)


            initial_residual = predicted_residual if use_prediction else r0
            # A jump/steep change in an applied command can place a nonlinear
            # flow equation at a kink (e.g. valve flow at zero pressure drop).
            # Symmetric differences then average different physical branches.
            # Select the feasible one-sided derivative *before* the solve, not
            # as a second-solver fallback or an equation-specific workaround.
            previous_controls = self._accepted_control_values
            changed_controls = (
                previous_controls is not None and
                self._controls_changed(previous_controls, self._control_values(cache))
            )
            # Once a time-commanded mode change has been encountered, keep
            # one-sided differences for the remainder of this transient solve.
            # The same commands can move through zero-flow/kink regions again.
            if changed_controls:
                self._encountered_nonsmooth_control = True
            if self._encountered_nonsmooth_control and least_squares_settings.jacobian_method == "3-point":
                self.work_statistics["one_sided_jacobian_steps"] += 1
            step_settings = (
                replace(least_squares_settings, jacobian_method="2-point")
                if (self._encountered_nonsmooth_control
                    and least_squares_settings.jacobian_method == "3-point")
                else least_squares_settings
            )
            if self._may_try_modified_newton(
                cache, step_settings, initial_guess, initial_residual,
            ):
                jacobian, jacobian_scale = self._initial_newton_jacobian(
                    cache, x0, accepted_time, dt, len(r0),
                    step_settings.jacobian_method,
                    probe_x=probe_x, probe_residual=probe_residual,
                    baseline_residual=r0,
                )
                if event_jacobian_seed is not None and jacobian is None:
                    seed_jac, seed_scale, _, _, _, _, _, seed_method, _, _, _ = event_jacobian_seed
                    if seed_method == step_settings.jacobian_method:
                        if probe_x is not None and probe_residual is not None:
                            delta = (probe_x - x0) / seed_scale
                            expected = seed_jac @ delta
                            actual = self._active_equation_scaling.residual(probe_residual-r0)
                            magnitude = max(float(np.linalg.norm(actual)), float(np.linalg.norm(expected)))
                            if magnitude and np.isfinite(magnitude):
                                err = float(np.linalg.norm(actual-expected)) / magnitude
                            else:
                                err = float("inf")
                            if np.isfinite(err) and err <= 0.65:
                                jacobian, jacobian_scale = seed_jac, seed_scale
                                self.work_statistics["event_jacobian_seed_uses"] += 1
                            else:
                                self.work_statistics["event_jacobian_seed_rejections"] += 1
                try:
                    newton_sol, _ = self._try_modified_newton(
                        cache, initial_guess, initial_residual, step_settings,
                        jacobian=jacobian, jacobian_scale=jacobian_scale,
                    )

                    # _try_modified_newton already performed the final state
                    # evaluation and residual verification at this candidate.
                    accepted_residual = np.asarray(newton_sol.fun, dtype=float)
                except SensorDataStop:
                    raise
                except Exception:
                    self.work_statistics["newton_rejections"] += 1
                    self._clear_newton_jacobian()
                    # A component can switch equations without a Sensor event.
                    # If its old row weights fail, rebuild from the new equations.
                    scaling = self._active_equation_scaling
                    if scaling is not None and scaling.reused:
                        scaling.scales = None
                        scaling.reused = False
                    self._restore_trial_start(
                        cache, accepted_snapshot, x0, t_new, dt,
                    )
                else:
                    elapsed_time = time.perf_counter() - start_time
                    self._finish_trial(
                        cache, x0, t_new, dt, len(accepted_residual), commit=commit,
                        jacobian=newton_sol.jacobian,
                        jacobian_scale=newton_sol.jacobian_scale,
                        jacobian_method=step_settings.jacobian_method,
                        jacobian_reused=newton_sol.jacobian_reused,
                        jacobian_rebuilt=newton_sol.jacobian_rebuilt,
                    )
                    newton_sol.fun = accepted_residual
                    newton_sol.cost = 0.5 * float(np.dot(accepted_residual, accepted_residual))
                    return self._make_diagnostics(
                        time=t_new, dt=dt, x0=initial_guess,
                        sol=newton_sol, residual=accepted_residual,
                        elapsed_time=elapsed_time,
                    )


            try:
                sol, stopped_early = self._solve_with_early_acceptance(
                    cache, initial_guess, step_settings,
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
                    cache, x0, step_settings,
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
            sol.scaled_fun = self._active_equation_scaling.residual(final_residual).copy()
            sol.fun = np.asarray(final_residual, dtype=float).copy()
            sol.cost = 0.5 * float(np.dot(sol.fun, sol.fun))
            elapsed_time = time.perf_counter() - start_time
            self._finish_trial(cache, x0, t_new, dt, len(final_residual), commit=commit)
            return self._make_diagnostics(
                time=t_new, dt=dt, x0=initial_guess, sol=sol,
                residual=final_residual, elapsed_time=elapsed_time,
            )

        except Exception:
            self._pending_trial = None
            self._clear_newton_jacobian()
            cache.clear_sequence_command_pending()
            cache.restore_mutable_states(accepted_snapshot)
            self.network.time.value = accepted_time
            raise
        finally:
            if commit or self._pending_trial is None:
                cache.clear_sequence_command_pending()
            self._active_cache = None
            self._active_equation_scaling = None

    def _least_squares_kwargs(
        self, cache, x0: np.ndarray, settings: LeastSquaresSettings,
    ) -> dict[str, Any]:
        """Construct the original scipy.optimize.least_squares arguments."""
        kwargs: dict[str, Any] = {
            "fun": (lambda x: self._active_equation_scaling.residual(self.residual(x)))
                   if self._active_equation_scaling is not None else self.residual,
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
        scaled_max = self._residual_norm(final_residual)
        if scaled_max <= settings.rtol:
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
            f"max |scaled residual| = {scaled_max:.3e}",
            f"per-step residual tolerance = {settings.rtol:.3e}",
        ]
        if worst_rows:
            lines.extend(["", "Largest residuals:"])
            lines.extend(f"  - {label}: {value:.6e}" for label, value in worst_rows)
        raise TransientStepError("\n".join(lines))
