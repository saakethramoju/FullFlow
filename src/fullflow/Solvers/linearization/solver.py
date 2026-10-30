from __future__ import annotations

from collections.abc import Sequence
from bisect import bisect_left
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve

from fullflow.System.State import is_assignable_state_like, is_state_like

from .operating_point import OperatingPointError, OperatingPointEvaluator, _number


class LinearizationError(OperatingPointError):
    """A consistent local linearization could not be calculated."""


@dataclass(frozen=True)
class LinearizationResult:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    x0: np.ndarray
    xdot0: np.ndarray
    u0: np.ndarray
    y0: np.ndarray
    states: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    time: float
    model_options: tuple[tuple[str, str | None], ...]
    network_name: str
    solver_type: str
    diagnostics: dict

    @property
    def jacobian(self):
        return self.D

    def save(self, filename):
        from .results import save_linearization
        return save_linearization(self, filename)

    def print(self, *, console=None):
        from .diagnostics import print_linearization
        print_linearization(self, console=console)


def _step(value, state):
    step = 1e-5 * max(abs(value), 1e-3)
    if isinstance(getattr(state, "value", None), bool):
        raise LinearizationError("Boolean States cannot be continuous linearization variables.")
    lower = float(getattr(state, "lower_bound", -np.inf))
    upper = float(getattr(state, "upper_bound", np.inf))
    if np.isfinite(lower):
        step = min(step, 0.49 * (value - lower))
    if np.isfinite(upper):
        step = min(step, 0.49 * (upper - value))
    if step <= max(1e-12 * abs(value), 1e-15):
        raise LinearizationError(f"Cannot perturb {getattr(state, 'label', state)} within its bounds.")
    return step


def _column(evaluate, value, state):
    step = _step(value, state)
    last_error = None
    for _ in range(5):
        try:
            plus = evaluate(value + step)
            minus = evaluate(value - step)
            derivative = (plus - minus) / (2 * step)
            half_plus = evaluate(value + 0.5 * step)
            half_minus = evaluate(value - 0.5 * step)
            refined = (half_plus - half_minus) / step
            error = np.abs(derivative - refined)
            scale = np.maximum(np.maximum(np.abs(derivative), np.abs(refined)), 1e-8)
            if np.any(error > 1e-6 + 5e-3 * scale):
                raise LinearizationError("Finite-difference derivatives are not consistent at this operating point.")
            return refined
        except Exception as error:
            last_error = error
            step *= 0.5
    raise LinearizationError(f"Perturbation failed for {getattr(state, 'label', state)}: {last_error}") from last_error


def _solve(matrix, right):
    if matrix.shape[0] == 0:
        return np.zeros((0, right.shape[1])), 1.0
    row = np.maximum(np.max(np.abs(matrix), axis=1), 1e-30)
    scaled = matrix / row[:, None]
    col = np.maximum(np.max(np.abs(scaled), axis=0), 1e-30)
    scaled /= col[None, :]
    condition = np.linalg.cond(scaled)
    if not np.isfinite(scaled).all() or not np.isfinite(condition) or condition > 1e12:
        raise LinearizationError("The state transformation or algebraic balances are singular.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            return solve(scaled, right / row[:, None], check_finite=True) / col[:, None], float(condition)
        except (ValueError, np.linalg.LinAlgError, LinAlgWarning) as error:
            raise LinearizationError(f"Sensitivity system could not be solved: {error}") from error


def _balance_tolerance(solver):
    """Use the tolerance under which the operating point was accepted."""
    tolerance = getattr(solver, "_linearization_tolerance", None)
    if tolerance is not None:
        return max(1e-6, float(tolerance))
    diagnostics = getattr(solver, "_last_solve_diagnostics", None)
    settings = getattr(diagnostics, "settings", None)
    return max(1e-6, float(getattr(settings, "rtol", 1e-6)))


def _display_label(value):
    return str(value).replace(":", ".", 1)


def linearize(
    solver, inputs: Sequence | None = None, outputs: Sequence | None = None,
    filename=None, *, times=None, verbose=False,
):
    """Linearize the last accepted operating point in physical state coordinates."""
    if times is not None:
        if not hasattr(solver, "step_solver"):
            raise LinearizationError("times is only supported by Transient.linearize().")
        return _linearize_many(solver, inputs, outputs, filename, times, verbose)
    if getattr(solver, "_linearization_issue", None) and hasattr(solver, "step_solver"):
        raise LinearizationError(solver._linearization_issue)
    try:
        point = OperatingPointEvaluator(solver)
    except OperatingPointError as error:
        raise LinearizationError(str(error)) from error
    if point.transient and any(getattr(item, "force_steady", False) for item in getattr(point.cache, "transient_items", ())):
        raise LinearizationError("Linearization of force-steady transient states is not supported.")
    inputs = () if inputs is None else tuple(inputs)
    outputs = () if outputs is None else tuple(outputs)
    if len({id(v) for v in inputs}) != len(inputs):
        raise LinearizationError("Inputs must be unique States.")
    if any(not is_assignable_state_like(v) for v in inputs):
        raise LinearizationError("Inputs must be independent, writable States.")
    if any(not is_state_like(v) and not callable(v) for v in outputs):
        raise LinearizationError("Outputs must be States or callables.")

    protected = {id(v) for v in point.variables + point.states}
    available = {id(v) for v in point.variables + point.states}
    available.update(id(v) for v in getattr(point.cache, "state_refs", ()))
    available.update(id(v) for v in getattr(point.cache, "all_state_refs", ()))
    available.update(
        id(tracked["value"])
        for tracked in solver.network.tracked_state_list
        if is_assignable_state_like(tracked.get("value"))
    )
    for state in inputs:
        if id(state) in protected or id(state) not in available:
            raise LinearizationError("Each input must be an independent State in the active network.")

    def label(state):
        if callable(state) and not is_state_like(state):
            return getattr(state, "__name__", "output")

        lookup = getattr(state, "lookup", None)
        attribute = getattr(state, "name", None)
        if lookup is not None and attribute:
            return f"{lookup.name}.{attribute}"

        for dynamic_state, dynamic_label in zip(point.states, point.state_labels):
            if dynamic_state is state:
                return _display_label(dynamic_label)

        names = list(getattr(state, "labels", ()))
        names.extend(point.cache.find_variable_labels(state))
        names = [
            name for name in names
            if "_attribute_cache" not in name and ".kwargs." not in name
        ]
        if names:
            names.sort(key=lambda name: ("[" in name, len(name)))
            return _display_label(names[0])
        return _display_label(getattr(state, "label", state))

    w0 = point.w0
    u0 = np.asarray([_number(item) for item in inputs])
    n, m, r = len(point.states), len(inputs), len(outputs)
    k = len(point.balances)
    count = n + n + k + r

    def sample(w=w0, u=u0):
        overrides = {state: float(value) for state, value in zip(inputs, u)}
        x, f, g, y = point.evaluate_raw(w, overrides, outputs)
        return np.concatenate((x, f, g, y))

    try:
        base = sample()
        if np.max(np.abs(base[2 * n:2 * n + k]), initial=0.0) > _balance_tolerance(solver):
            raise LinearizationError("The operating point does not satisfy its algebraic balances.")
        if not point.transient and np.max(np.abs(base[n:2 * n]), initial=0.0) > _balance_tolerance(solver):
            raise LinearizationError("The steady-state dynamic derivatives are not zero. Solve again before linearizing.")
        if not np.allclose(base[:n], point.x0, rtol=1e-8, atol=1e-9):
            raise LinearizationError("Physical states do not match the solved operating point.")

        derivatives_w = np.empty((count, len(w0)))
        for i, variable in enumerate(point.variables):
            derivatives_w[:, i] = _column(
                lambda value, i=i: sample(np.array([value if j == i else w for j, w in enumerate(w0)])),
                w0[i], variable,
            )

        derivatives_u = np.empty((count, m))
        for i, state in enumerate(inputs):
            derivatives_u[:, i] = _column(
                lambda value, i=i: sample(u=np.array([value if j == i else u for j, u in enumerate(u0)])),
                u0[i], state,
            )

        Xw, Fw = derivatives_w[:n], derivatives_w[n:2 * n]
        Gw, Yw = derivatives_w[2 * n:2 * n + k], derivatives_w[2 * n + k:]
        Xu, Fu = derivatives_u[:n], derivatives_u[n:2 * n]
        Gu, Yu = derivatives_u[2 * n:2 * n + k], derivatives_u[2 * n + k:]

        system = np.vstack((Xw, Gw))
        if system.shape[0] != system.shape[1]:
            raise LinearizationError("The physical states and algebraic balances do not form a square system.")

        dx = np.vstack((np.eye(n), np.zeros((k, n))))
        du = -np.vstack((Xu, Gu))
        transform, condition = _solve(system, np.hstack((dx, du)))
        wx, wu = transform[:, :n], transform[:, n:]
        result = LinearizationResult(
            A=Fw @ wx,
            B=Fu + Fw @ wu,
            C=Yw @ wx,
            D=Yu + Yw @ wu,
            x0=base[:n].copy(),
            xdot0=base[n:2 * n].copy(),
            u0=u0.copy(),
            y0=base[2 * n + k:].copy(),
            states=tuple(_display_label(name) for name in point.state_labels),
            inputs=tuple(label(state) for state in inputs),
            outputs=tuple(label(state) for state in outputs),
            time=point.time,
            model_options=point.models,
            network_name=solver.network.name,
            solver_type="transient" if point.transient else "steady_state",
            diagnostics={
                "max_abs_balance_residual": float(np.max(np.abs(base[2 * n:2 * n + k]), initial=0.0)),
                "max_abs_steady_derivative": float(np.max(np.abs(base[n:2 * n]), initial=0.0)) if not point.transient else 0.0,
                "sensitivity_condition_number": condition,
            },
        )
        if filename is not None:
            if not str(filename).strip():
                raise LinearizationError("filename must not be empty.")
            result.save(filename)
        if verbose:
            from .diagnostics import print_linearization
            print_linearization(result, console=getattr(solver, "console", None))
        return result
    except LinearizationError:
        raise
    except OperatingPointError as error:
        raise LinearizationError(str(error)) from error
    except Exception as error:
        raise LinearizationError(f"Linearization failed: {error}") from error
    finally:
        point.snapshot.restore()


def _linearize_many(solver, inputs, outputs, filename, times, verbose=False):
    from .results import save_linearizations

    current = solver._linearization_source
    if current is None:
        raise LinearizationError("A successful transient solve is required.")
    try:
        current.validate()
    except ValueError as error:
        raise LinearizationError(str(error)) from error

    requested = tuple(float(t) for t in times)
    if not requested or len(set(requested)) != len(requested) or not np.isfinite(requested).all():
        raise LinearizationError("times must contain distinct, finite operating-point times.")

    accepted = sorted(solver._linearization_points)
    checkpoints = {}
    for time_value in requested:
        index = bisect_left(accepted, time_value)
        neighbors = accepted[max(0, index - 1):index + 1]
        if not neighbors:
            raise LinearizationError("No accepted operating-point checkpoints are available.")
        closest = min(neighbors, key=lambda t: abs(t - time_value))
        position = bisect_left(accepted, closest)
        distances = []
        if position > 0:
            distances.append(closest - accepted[position - 1])
        if position + 1 < len(accepted):
            distances.append(accepted[position + 1] - closest)
        tolerance = 1e-9 * max(1.0, abs(time_value))
        if distances:
            tolerance = min(tolerance, 0.25 * min(distances))
        if abs(closest - time_value) > tolerance:
            raise LinearizationError(
                f"No accepted operating point at t={time_value:g}. "
                f"Closest accepted time is t={closest:.12g} "
                f"({len(accepted)} checkpoints, from t={accepted[0]:.12g} "
                f"to t={accepted[-1]:.12g}). "
                "Requested times must match accepted solver timesteps; "
                "no interpolation is performed."
            )
        checkpoints[time_value] = solver._linearization_points[closest]

    for time_value, (_, issue) in checkpoints.items():
        if issue:
            raise LinearizationError(f"Cannot linearize a switching event at t={time_value:g}: {issue}")

    results = {}
    try:
        for time_value, (checkpoint, _) in checkpoints.items():
            checkpoint.restore()
            solver._linearization_source = checkpoint
            solver._linearization_issue = None
            results[time_value] = linearize(
                solver,
                inputs=inputs,
                outputs=outputs,
                verbose=verbose,
            )
        if filename is not None:
            save_linearizations(results, filename, solver.network.name)
        return results
    finally:
        current.restore()
        solver._linearization_source = current
        solver._linearization_issue = solver._linearization_points.get(current.time, (None, None))[1]
