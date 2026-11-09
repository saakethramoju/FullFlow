"""Localized switching of explicitly event-driven Sequence functions."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from fullflow.Exceptions import TransientStepError
from fullflow.Solvers.equation_scaling import EquationScaling
from fullflow.Solvers.linearization.checkpoints import OperatingCheckpoint


class SequenceSwitches:
    """Probe discontinuous controllers outside the implicit residual evaluation."""

    def __init__(self, solver):
        self.solver = solver
        self.sequences = tuple(
            component for component in solver.network.component_list
            if getattr(component, '_event_enabled', False)
        )
        self.records = []
        self.probes = 0
        self.reconciliation_residual_calls = 0
        self.reconciliation_scipy_solves = 0
        self.reconciliation_reported_jacobian_builds = 0
        self.reconciliation_scale_jacobian_builds = 0

    def initialize(self):
        for sequence in self.sequences:
            sequence.initialize_event()

    @staticmethod
    def _changed(a, b):
        return not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)

    def _requests(self, time_value):
        return [
            (sequence, requested)
            for sequence in self.sequences
            if self._changed(
                requested := sequence.event_request(time_value), sequence._event_command
            )
        ]

    def _guards(self):
        return {sequence: sequence._event_probe_inputs() for sequence in self.sequences}

    def _predict_time(self, left, right, t_left, t_right):
        """Estimate a switching instant using cheap pure-function evaluations."""
        fractions = []
        for sequence, a in left.items():
            b = right.get(sequence)
            if b is None or len(a) != len(b):
                continue
            try:
                x0 = tuple(float(value) for value in a)
                x1 = tuple(float(value) for value in b)
                if not all(math.isfinite(x) for x in (*x0, *x1)):
                    continue
                if self._changed(sequence._event_request_from_inputs(t_left, x0), sequence._event_command):
                    continue
                if not self._changed(sequence._event_request_from_inputs(t_right, x1), sequence._event_command):
                    continue
                low, high = 0.0, 1.0
                for _ in range(24):
                    fraction = (low + high) * 0.5
                    values = tuple(x + fraction * (y-x) for x, y in zip(x0, x1))
                    if self._changed(
                        sequence._event_request_from_inputs(t_left + fraction*(t_right-t_left), values),
                        sequence._event_command,
                    ):
                        high = fraction
                    else:
                        low = fraction
                fractions.append(high)
            except (ValueError, TypeError, OverflowError):
                continue
        fraction = min(fractions) if fractions else 0.5
        return t_left + min(0.99, max(0.01, fraction)) * (t_right-t_left)

    def _restore(self, checkpoint):
        checkpoint.restore()
        self.solver.step_solver.discard_trial()
        self.solver._cache().clear_sequence_command_pending()

    def _reconcile(self, cache, t, settings, state_settings, residual_count):
        """Rebalance algebraic unknowns at a switch without advancing dynamics."""
        if not cache.is_current():
            raise TransientStepError('A Sequence switch changed the network equation structure.')
        algebraic = tuple(
            state for state in cache.iteration_variables
            if id(state) not in cache.transient_ids
        )
        dynamic = tuple((state, float(state.value)) for state in cache.transient_variables)
        cache.set_transient_context(dt=0.0)
        if not algebraic:
            cache.apply_sequence_commands(t)
            self.solver.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance, cache=cache,
            )
            if any(float(state.value) != old for state, old in dynamic):
                raise TransientStepError('Sequence switch unexpectedly modified differential states.')
            cache.store_previous_values()
            self.solver.step_solver.discard_trial()
            return

        def residual(values):
            self.reconciliation_residual_calls += 1
            for state, value in zip(algebraic, values):
                state.value = float(value)
            cache.apply_sequence_commands(t)
            self.solver.evaluator.run(
                max_passes=state_settings.max_passes,
                tolerance=state_settings.tolerance, cache=cache,
            )
            r = cache.collect_residuals(dt=1.0)
            if len(r) != residual_count:
                raise TransientStepError('Sequence switch changed the number of equations.')
            return r[len(cache.transient_items):]

        current = np.asarray([float(s.value) for s in algebraic])
        remaining = residual(current)
        if not np.all(np.isfinite(remaining)):
            raise TransientStepError(f'Nonfinite algebraic residual at Sequence switch t={t:.9g}.')
        scales = None
        if remaining.size and np.max(np.abs(remaining)) > settings.rtol:
            lower = np.asarray([float(s.lower_bound) for s in algebraic])
            upper = np.asarray([float(s.upper_bound) for s in algebraic])
            scales = EquationScaling(np.ones(len(remaining), dtype=bool))
            self.reconciliation_scale_jacobian_builds += 1
            scales.initialize(
                residual, current, settings.jacobian_method,
                bounds=(lower, upper) if settings.solver_method != 'lm' else None,
            )
            remaining = residual(current)
            if np.max(np.abs(scales.residual(remaining)), initial=0.0) > settings.rtol:
                for _ in range(3):
                    current = np.asarray([float(s.value) for s in algebraic])
                    bounds = {} if settings.solver_method == 'lm' else {
                        'bounds': (lower-current, upper-current),
                    }
                    self.reconciliation_scipy_solves += 1
                    sol = least_squares(
                        lambda correction: scales.residual(residual(current+correction)),
                        np.zeros_like(current), jac=settings.jacobian_method,
                        method=settings.solver_method, **bounds,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100,
                    )
                    self.reconciliation_reported_jacobian_builds += int(sol.njev or 0)
                    remaining = residual(current+sol.x)
                    if (np.all(np.isfinite(remaining)) and
                            np.max(np.abs(scales.residual(remaining)), initial=0.0) <= settings.rtol):
                        break
        error = np.max(np.abs(scales.residual(remaining) if scales is not None else remaining), initial=0.0)
        if not np.isfinite(error) or error > settings.rtol:
            labels = cache.collect_residual_labels()[len(cache.transient_items):]
            detail = ', '.join(f'{name}={value:.4g}' for name, value in
                               sorted(zip(labels, remaining), key=lambda p: -abs(p[1]))[:5])
            raise TransientStepError(
                f'Sequence switch algebraic equations did not converge at t={t:.9g}: '
                f'max residual={error:.3e}; {detail}'
            )
        if any(float(s.value) != old for s, old in dynamic):
            raise TransientStepError('Sequence switch unexpectedly modified differential states.')
        cache.store_previous_values()
        self.solver.step_solver.discard_trial()

    def _accept_switches(self, switches, event_time, evaluation_time, settings, state_settings):
        """Commit simultaneous commands and reconcile at one physical instant."""
        previous = [(sequence, float(sequence._event_command), value) for sequence, value in switches]
        for sequence, _, value in previous:
            sequence.accept_event(value)
        self._reconcile(
            self.solver._cache(), evaluation_time, settings, state_settings,
            len(self.solver._cache().collect_residual_labels()),
        )
        for sequence, old_value, value in previous:
            self.records.append({
                'time': float(event_time), 'sequence': sequence.name,
                'previous_value': old_value, 'value': float(value),
            })
        self.solver.step_solver.invalidate_equation_scales()

    def _apply_boundary_switches(self, checkpoint, time_value, probe_time, settings, state_settings):
        """Check a one-sided command change without integrating the plant."""
        self._restore(checkpoint)
        cache = self.solver._cache()
        self.solver.network.time.value = probe_time
        cache.set_transient_context(dt=0.0)
        cache.apply_sequence_commands(probe_time)
        self.solver.evaluator.run(
            max_passes=state_settings.max_passes,
            tolerance=state_settings.tolerance, cache=cache,
        )
        switches = self._requests(probe_time)
        if not switches:
            self._restore(checkpoint)
            return False
        self._accept_switches(switches, time_value, probe_time, settings, state_settings)
        self.solver.network.time.value = time_value
        return True

    def run_step(self, t_new, dt, settings, state_settings, _boundary_passes=0):
        if not self.sequences:
            return self.solver.step_solver.run_once(
                t_new=t_new, dt=dt,
                least_squares_settings=settings, state_settings=state_settings,
            ), False

        if _boundary_passes >= 32:
            raise TransientStepError(
                'More than 32 simultaneous or repeated Sequence switches at one timestep boundary.'
            )
        cache = self.solver._cache()
        original = OperatingCheckpoint(self.solver.network, cache)
        initial_record_count = len(self.records)
        t_start = original.time
        try:
            # An already-due switch is a zero-time mode change, not a transient step.
            if self._requests(t_start):
                if not self._apply_boundary_switches(
                    original, t_start, t_start, settings, state_settings,
                ):
                    raise TransientStepError(
                        f'Sequence command changed during the boundary check at t={t_start:.9g}.'
                    )
                diagnostics, _ = self.run_step(
                    t_new, dt, settings, state_settings, _boundary_passes + 1,
                )
                return diagnostics, True

            start_x = cache.iteration_value_array()
            start_guards = self._guards()
            diagnostics = self.solver.step_solver.run_once(
                t_new=t_new, dt=dt, least_squares_settings=settings,
                state_settings=state_settings, commit=False,
            )
            if not self._requests(t_new):
                self.solver.step_solver.accept_trial()
                return diagnostics, False
            end_guards = self._guards()
            end_x = cache.iteration_value_array()
            prior_jacobian = self.solver.step_solver.event_jacobian_seed()
            self._restore(original)
            low, high = t_start, t_new
            low_x, high_x = start_x, end_x
            low_guards, high_guards = start_guards, end_guards
            tolerance = max(1e-10, abs(dt) * 1e-6)
            # Probe the right limit at the frozen accepted plant state first.
            # A known-time setpoint jump must not enter the integration search.
            boundary_time = min(t_new, t_start + tolerance)
            if boundary_time > t_start and self._apply_boundary_switches(
                original, t_start, boundary_time, settings, state_settings,
            ):
                diagnostics, _ = self.run_step(
                    t_new, dt, settings, state_settings, _boundary_passes + 1,
                )
                return diagnostics, True
            for iteration in range(40):
                if high - low <= tolerance:
                    break
                candidate_time = (
                    self._predict_time(low_guards, high_guards, low, high)
                    if iteration < 8 else 0.5 * (low+high)
                )
                if not low < candidate_time < high:
                    break
                self._restore(original)
                self.probes += 1
                fraction = (candidate_time - low) / (high - low)
                guess = low_x + fraction * (high_x - low_x)
                self.solver.step_solver.run_once(
                    t_new=candidate_time, dt=candidate_time-t_start,
                    least_squares_settings=settings,
                    state_settings=state_settings, commit=False,
                    trial_guess=guess,
                    event_jacobian_seed=prior_jacobian,
                )
                prior_jacobian = self.solver.step_solver.event_jacobian_seed()
                guards = self._guards()
                candidate_x = cache.iteration_value_array()
                if self._requests(candidate_time):
                    high, high_guards, high_x = candidate_time, guards, candidate_x
                else:
                    low, low_guards, low_x = candidate_time, guards, candidate_x

            # A jump in external time-dependent commands can be bracketed right
            # beside an accepted step. Confirm it with frozen plant states before
            # treating it as a zero-duration event.
            if high - t_start <= tolerance and self._apply_boundary_switches(
                original, t_start, high, settings, state_settings,
            ):
                diagnostics, _ = self.run_step(
                    t_new, dt, settings, state_settings, _boundary_passes + 1,
                )
                return diagnostics, True

            self._restore(original)
            diagnostics = self.solver.step_solver.run_once(
                t_new=high, dt=high-t_start,
                least_squares_settings=settings,
                state_settings=state_settings, commit=False,
                trial_guess=high_x,
                event_jacobian_seed=prior_jacobian,
            )
            switches = self._requests(high)
            if not switches:
                raise TransientStepError(f'Could not reproduce Sequence switch at t={high:.9g}.')
            self.solver.step_solver.accept_trial()
            previous = [(sequence, float(sequence._event_command), value) for sequence, value in switches]
            for sequence, _, value in previous:
                sequence.accept_event(value)
            self._reconcile(self.solver._cache(), high, settings, state_settings, len(diagnostics.residual))
            for sequence, old_value, value in previous:
                self.records.append({
                    'time': float(high), 'sequence': sequence.name,
                    'previous_value': old_value, 'value': float(value),
                })
            self.solver.step_solver.invalidate_equation_scales()
            return diagnostics, True
        except Exception:
            del self.records[initial_record_count:]
            self._restore(original)
            raise
