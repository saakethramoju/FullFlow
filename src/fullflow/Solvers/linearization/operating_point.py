from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from fullflow.Solvers.equations import (
    component_dynamics,
    component_balances,
    balance_object_equations,
)
from fullflow.Solvers.transient.models import NetworkStateSnapshot
from fullflow.System.State import is_assignable_state_like, is_state_like


class OperatingPointError(RuntimeError):
    """An operating point cannot be evaluated consistently."""


def _number(value):
    if is_state_like(value):
        value = value.value
    elif callable(value):
        value = value()
    result = float(value)
    if not np.isfinite(result):
        raise OperatingPointError(f"Non-finite equation value: {result!r}.")
    return result


from .checkpoints import OperatingCheckpoint as EvaluationSnapshot


@dataclass(frozen=True)
class OperatingPointResult:
    time: float
    states: tuple[str, ...]
    algebraic: tuple[str, ...]
    x: np.ndarray
    xdot: np.ndarray
    z: np.ndarray
    residuals: np.ndarray
    outputs: np.ndarray
    model_options: tuple[tuple[str, str | None], ...]


class OperatingPointEvaluator:
    """Evaluate frozen storage states with active algebraic balances satisfied."""

    def __init__(self, solver):
        self.solver = solver
        self.network = solver.network
        self.transient = hasattr(solver, "step_solver")
        self._check_source()
        self.cache = solver._cache()
        self.version = self.network.version
        self.components = tuple(self.cache.active_component_list if not self.transient else self.cache.component_list)
        self.models = self._model_options()
        self.dynamic = tuple(
            eq for component in self.components for eq in component_dynamics(component)
        )
        self.balances = tuple(
            eq for component in self.components
            for eq in self._component_balances(component)
        ) + tuple(
            eq for balance in self.cache.balance_list
            for eq in balance_object_equations(balance)
        )
        self.variables = tuple(eq.variable for eq in self.dynamic + self.balances)
        self.states = tuple(eq.state for eq in self.dynamic)
        self.state_labels = tuple(self.cache.state_label(eq.owner, eq.state) for eq in self.dynamic)
        self.balance_labels = tuple(self.cache.state_label(eq.owner, eq.variable) for eq in self.balances)
        self._validate_equations()
        self._equation_keys = tuple(
            (id(eq.owner), id(eq.variable), id(eq.state) if hasattr(eq, "state") else None)
            for eq in self.dynamic + self.balances
        )
        self._trial_snapshot = None
        self._trial_inputs = {}
        self.snapshot = EvaluationSnapshot(self.network, self.cache)
        self.mode_signature = self.snapshot.modes
        try:
            self.solver.evaluator.run(max_passes=12, tolerance=1e-10)
            if self.snapshot.mode_signature() != self.mode_signature:
                raise OperatingPointError("An operating mode changed during baseline evaluation.")
            self.x0 = np.array([_number(state) for state in self.states], dtype=float)
            self.w0 = np.array([_number(variable) for variable in self.variables], dtype=float)
            self.time = float(self.network.time.value)
        finally:
            self.snapshot.restore()

    def _check_source(self):
        source = getattr(self.solver, "_linearization_source", None)
        if source is None:
            raise OperatingPointError("A successful solve is required before linearization.")
        try:
            source.validate()
        except ValueError as error:
            raise OperatingPointError(str(error)) from error
        if self.transient:
            rows = self.solver.step_diagnostics
            if not rows:
                raise OperatingPointError("Transient requires an accepted timestep before analysis.")
        elif getattr(self.solver, "_last_success_kind", None) not in ("solve", "forced_steady_time_sweep"):
            raise OperatingPointError("SteadyState requires a successful nonlinear solve before analysis.")

    def _model_options(self):
        return tuple(
            (model.name, model.active_option_name)
            for model in self.network.model_list
        )

    def _component_balances(self, component):
        method = getattr(self.cache, "_component_balance_equations", None)
        return method(component) if method is not None else component_balances(component)

    def _validate_equations(self):
        ids = [id(variable) for variable in self.variables]
        if len(ids) != len(set(ids)):
            raise OperatingPointError("Dynamic and algebraic equations share a solve variable.")
        if len({id(state) for state in self.states}) != len(self.states):
            raise OperatingPointError("The same conserved state is integrated more than once.")
        for variable in self.variables:
            if not is_assignable_state_like(variable):
                raise OperatingPointError("Operating-point solve variables must be writable States.")
        if set(ids) != {id(v) for v in self.cache.iteration_variables}:
            raise OperatingPointError("Active equation variables do not match the solver's iteration variables.")

    def _check_configuration(self):
        if self.network.version != self.version or self._model_options() != self.models:
            raise OperatingPointError("Network structure or model option changed. Solve again before analysis.")
        if tuple(self.cache.iteration_variables) != self.variables:
            raise OperatingPointError("Solver equation ordering changed. Solve again before analysis.")

    def _equations(self):
        dynamics = tuple(eq for c in self.components for eq in component_dynamics(c))
        balances = tuple(eq for c in self.components for eq in self._component_balances(c)) + tuple(
            eq for b in self.cache.balance_list for eq in balance_object_equations(b)
        )
        keys = tuple(
            (id(eq.owner), id(eq.variable), id(eq.state) if hasattr(eq, "state") else None)
            for eq in dynamics + balances
        )
        if keys != self._equation_keys:
            raise OperatingPointError("The active equation set changed during evaluation.")
        return dynamics, balances

    def evaluate_raw(self, variables=None, inputs=None, outputs=None):
        """Evaluate physical states, derivatives, balances and outputs at fixed coordinates."""
        self._check_configuration()
        variables = self.w0 if variables is None else np.asarray(variables, dtype=float)
        if variables.shape != self.w0.shape:
            raise OperatingPointError("Invalid iteration-coordinate vector.")
        inputs = {} if inputs is None else dict(inputs)
        outputs = () if outputs is None else tuple(outputs)
        try:
            self.snapshot.restore()
            for state, value in inputs.items():
                state.value = float(value)
            for state, value in zip(self.variables, variables):
                state.value = float(value)
            self.solver.evaluator.run(max_passes=12, tolerance=1e-10)
            for state, value in inputs.items():
                if not np.isclose(_number(state), float(value), rtol=1e-12, atol=0):
                    raise OperatingPointError("An evaluated component overwrote a prescribed input.")
            dynamics, balances = self._equations()
            if self.snapshot.mode_signature() != self.mode_signature:
                raise OperatingPointError("An operating mode changed during a perturbation.")
            return (
                np.asarray([_number(eq.state) for eq in dynamics]),
                np.asarray([_number(eq.derivative) for eq in dynamics]),
                np.asarray([_number(eq.residual) for eq in balances]),
                np.asarray([_number(item) for item in outputs]),
            )
        except OperatingPointError:
            raise
        except Exception as error:
            raise OperatingPointError(f"Equation evaluation failed: {error}") from error
        finally:
            self.snapshot.restore()

    def _residual(self, w, x):
        self._trial_snapshot.restore()
        for state, value in zip(self.variables, w):
            state.value = float(value)
        self.solver.evaluator.run(max_passes=12, tolerance=1e-10)
        for state, expected in self._trial_inputs.items():
            if _number(state) != expected:
                raise OperatingPointError("An evaluated component overwrote a prescribed input.")
        dynamics, balances = self._equations()
        if self.snapshot.mode_signature() != self.mode_signature:
            raise OperatingPointError("An operating mode changed during a perturbation.")
        stored = [(_number(eq.state) - target) / max(abs(target), 1.0) for eq, target in zip(dynamics, x)]
        residuals = [_number(eq.residual) for eq in balances]
        return np.asarray(stored + residuals, dtype=float)

    def evaluate(
        self,
        states: Sequence[float] | None = None,
        inputs: Mapping | None = None,
        outputs: Sequence | None = None,
    ) -> OperatingPointResult:
        """Resolve algebraic variables without advancing the conserved states."""
        self._check_configuration()
        target = np.array(self.x0 if states is None else states, dtype=float)
        if target.shape != self.x0.shape or not np.all(np.isfinite(target)):
            raise OperatingPointError("Invalid conserved-state vector.")
        inputs = {} if inputs is None else dict(inputs)
        outputs = () if outputs is None else tuple(outputs)
        if any(not is_assignable_state_like(state) for state in inputs):
            raise OperatingPointError("Inputs must be independent writable States.")
        registered = {id(s) for s in self.cache.iteration_variables}
        registered.update(id(s) for s in getattr(self.cache, "state_refs", ()))
        registered.update(id(s) for s in self.states)
        if any(id(state) not in registered for state in inputs):
            raise OperatingPointError("Every input must belong to the active network.")
        if any(id(state) in {id(v) for v in self.variables + self.states} for state in inputs):
            raise OperatingPointError("An input cannot also be a dynamic state or solver variable.")
        try:
            self.snapshot.restore()
            for state, value in inputs.items():
                state.value = float(value)
            self._trial_snapshot = EvaluationSnapshot(self.network, self.cache)
            self._trial_inputs = {state: _number(state) for state in inputs}
            w0 = self.w0.copy()
            baseline_residuals = self._residual(w0, target)
            if len(baseline_residuals) != len(w0):
                raise OperatingPointError("Frozen operating point requires matching numbers of equations and variables.")
            bounds = (
                np.asarray([v.lower_bound for v in self.variables], dtype=float),
                np.asarray([v.upper_bound for v in self.variables], dtype=float),
            )
            if len(w0):
                if np.any(w0 < bounds[0]) or np.any(w0 > bounds[1]):
                    raise OperatingPointError("Current solve variables violate their bounds.")
                sol = least_squares(
                    lambda w: self._residual(w, target),
                    w0, bounds=bounds, method="trf", jac="3-point",
                    ftol=1e-12, xtol=1e-12, gtol=1e-12,
                    max_nfev=200,
                )
                if not sol.success:
                    raise OperatingPointError(f"Algebraic solve failed: {sol.message}")
                residuals = self._residual(sol.x, target)
            else:
                residuals = self._residual([], target)
            if np.any(np.abs(residuals) > 1e-6):
                raise OperatingPointError(f"Operating-point residual too large: {np.max(np.abs(residuals)):.4g}.")
            dynamics, balances = self._equations()
            if self.snapshot.mode_signature() != self.mode_signature:
                raise OperatingPointError("An operating mode changed during a perturbation.")
            derivatives = np.asarray([_number(eq.derivative) for eq in dynamics], dtype=float)
            values = np.asarray([_number(item) for item in outputs], dtype=float)
            z = np.asarray([_number(eq.variable) for eq in balances], dtype=float)
            return OperatingPointResult(
                self.time, self.state_labels, self.balance_labels,
                target.copy(), derivatives, z, residuals.copy(), values, self.models,
            )
        except OperatingPointError:
            raise
        except Exception as error:
            raise OperatingPointError(f"Operating-point evaluation failed: {error}") from error
        finally:
            self.snapshot.restore()
            self._trial_snapshot = None
            self._trial_inputs = {}
