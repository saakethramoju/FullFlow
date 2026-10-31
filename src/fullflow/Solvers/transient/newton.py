"""Safeguarded Newton solver with Broyden Jacobian updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import linalg
from scipy.optimize import OptimizeResult


@dataclass(frozen=True, slots=True)
class ModifiedNewtonOptions:
    max_iterations: int = 14
    max_jacobian_builds: int = 3
    max_backtracks: int = 9
    max_relative_evaluations: int = 4
    refresh_ratio: float = 0.80
    sufficient_decrease: float = 1.0e-4
    secant_error_limit: float = 0.65


class ModifiedNewtonFailure(RuntimeError):
    """Signal an unsuccessful nonlinear solve."""


def solve_modified_newton(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    f0: np.ndarray,
    *,
    rtol: float,
    jacobian_method: str = "2-point",
    options: ModifiedNewtonOptions | None = None,
    initial_jacobian: np.ndarray | None = None,
    initial_scale: np.ndarray | None = None,
) -> OptimizeResult:
    """Solve an unbounded system using modified Newton and Broyden updates."""
    opt = options or ModifiedNewtonOptions()
    x0 = np.asarray(x0, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    if x0.ndim != 1 or f0.ndim != 1 or not len(x0) or len(f0) < len(x0):
        raise ModifiedNewtonFailure("Modified Newton requires m >= n > 0.")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(f0)):
        raise ModifiedNewtonFailure("Modified Newton requires finite initial values.")
    if jacobian_method not in {"2-point", "3-point"}:
        raise ModifiedNewtonFailure("Unsupported numerical Jacobian method.")
    if rtol <= 0.0 or not np.isfinite(rtol):
        raise ModifiedNewtonFailure("Invalid residual tolerance.")

    scale = np.maximum(np.abs(x0), 1.0)
    y = x0 / scale
    f = f0.copy()
    count = 0
    builds = 0
    updates = 0
    total_iterations = 0
    matrix = None
    reused_jacobian = False
    last_secant_error = None
    eps = np.cbrt(np.finfo(float).eps) if jacobian_method == "3-point" else np.sqrt(np.finfo(float).eps)
    max_evals = max(20, opt.max_relative_evaluations * (len(x0) + 1))

    if initial_jacobian is not None and initial_scale is not None:
        prior_matrix = np.asarray(initial_jacobian, dtype=float)
        prior_scale = np.asarray(initial_scale, dtype=float)
        if (prior_matrix.shape == (len(f), len(y))
                and prior_scale.shape == scale.shape
                and np.all(np.isfinite(prior_matrix))
                and np.all(np.isfinite(prior_scale))
                and np.all(prior_scale > 0.0)):
            ratio = scale / prior_scale
            if np.all(np.isfinite(ratio)) and np.all((ratio >= 0.2) & (ratio <= 5.0)):
                matrix = prior_matrix * ratio[np.newaxis, :]
                if np.all(np.isfinite(matrix)):
                    reused_jacobian = True
                else:
                    matrix = None

    def evaluate(candidate_y: np.ndarray) -> np.ndarray:
        """Evaluate the residual at a candidate Newton iterate."""
        nonlocal count
        if count >= max_evals:
            raise ModifiedNewtonFailure("Modified Newton evaluation budget exhausted.")
        candidate_x = candidate_y * scale
        if not np.all(np.isfinite(candidate_x)):
            raise ModifiedNewtonFailure("Non-finite trial unknowns.")
        values = np.asarray(fun(candidate_x), dtype=float)
        count += 1
        if values.shape != f0.shape or not np.all(np.isfinite(values)):
            raise ModifiedNewtonFailure("Non-finite or inconsistent trial residuals.")
        return values

    def build_jacobian(current_y: np.ndarray, current_f: np.ndarray) -> np.ndarray:
        """Build a finite-difference Newton Jacobian."""
        nonlocal builds
        if builds >= opt.max_jacobian_builds:
            raise ModifiedNewtonFailure("Jacobian refresh limit reached.")
        builds += 1
        jac = np.empty((len(current_f), len(current_y)), dtype=float)
        for j in range(len(current_y)):
            step = eps * max(abs(float(current_y[j])), 1.0)
            positive = current_y.copy()
            positive[j] += step
            positive_f = evaluate(positive)
            if jacobian_method == "3-point":
                negative = current_y.copy()
                negative[j] -= step
                negative_f = evaluate(negative)
                jac[:, j] = (positive_f - negative_f) / (2.0 * step)
            else:
                jac[:, j] = (positive_f - current_f) / step
        if not np.all(np.isfinite(jac)):
            raise ModifiedNewtonFailure("Non-finite numerical Jacobian.")
        return jac

    for iteration in range(opt.max_iterations):
        total_iterations = iteration + 1
        if float(np.max(np.abs(f))) <= rtol:
            break
        if matrix is None:
            matrix = build_jacobian(y, f)

        try:
            column_norm = np.max(np.abs(matrix), axis=0)
            positive_norm = column_norm[column_norm > 0.0]
            if not len(positive_norm):
                raise linalg.LinAlgError("Zero Jacobian")
            reference = float(np.median(positive_norm))
            column_scale = np.clip(
                reference / np.maximum(column_norm, np.finfo(float).tiny),
                1.0e-8, 1.0e8,
            )
            scaled_matrix = matrix * column_scale[np.newaxis, :]
            if not np.all(np.isfinite(scaled_matrix)):
                raise linalg.LinAlgError("Non-finite scaled Jacobian")
            if len(f) == len(y):
                with np.errstate(over="raise", invalid="raise"):
                    scaled_direction = linalg.solve(
                        scaled_matrix, -f, check_finite=False, assume_a="gen",
                    )
            else:
                scaled_direction, _, rank, _ = linalg.lstsq(
                    scaled_matrix, -f, cond=1.0e-12, check_finite=False,
                    lapack_driver="gelsy",
                )
                if rank < len(y):
                    raise linalg.LinAlgError("Rank-deficient Jacobian")
            direction = column_scale * scaled_direction
        except (linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
            matrix = None
            if builds >= opt.max_jacobian_builds:
                raise ModifiedNewtonFailure("Cannot solve Newton direction.")
            continue

        if not np.all(np.isfinite(direction)):
            raise ModifiedNewtonFailure("Non-finite Newton direction.")
        if not np.any(direction):
            matrix = None
            continue

        old_norm = float(np.linalg.norm(f))
        factor = 1.0
        accepted = False
        for _ in range(opt.max_backtracks):
            trial_y = y + factor * direction
            trial_f = evaluate(trial_y)
            trial_norm = float(np.linalg.norm(trial_f))
            if (float(np.max(np.abs(trial_f))) <= rtol or
                    trial_norm < old_norm * (1.0 - opt.sufficient_decrease * factor)):
                accepted = True
                break
            factor *= 0.5

        if not accepted:
            matrix = None
            if builds >= opt.max_jacobian_builds:
                raise ModifiedNewtonFailure("Backtracking did not reduce the residual.")
            continue

        dy = trial_y - y
        df = trial_f - f
        predicted_df = matrix @ dy
        prediction_scale = max(float(np.linalg.norm(df)), float(np.linalg.norm(predicted_df)))
        if prediction_scale > 0.0 and np.isfinite(prediction_scale):
            last_secant_error = float(np.linalg.norm(df - predicted_df)) / prediction_scale
        else:
            last_secant_error = None
        y, f = trial_y, trial_f
        converged = float(np.max(np.abs(f))) <= rtol
        denom = float(np.dot(dy, dy))
        if (not np.isfinite(denom) or denom <= 1.0e-24
                or last_secant_error is None
                or not np.isfinite(last_secant_error)
                or last_secant_error > opt.secant_error_limit):
            matrix = None
        else:
            matrix += np.outer(df - predicted_df, dy) / denom
            updates += 1
            if not np.all(np.isfinite(matrix)) or float(np.linalg.norm(f)) > opt.refresh_ratio * old_norm:
                matrix = None
        if converged:
            break
    else:
        raise ModifiedNewtonFailure("Modified Newton iteration limit reached.")

    if not np.all(np.isfinite(f)) or float(np.max(np.abs(f))) > rtol:
        raise ModifiedNewtonFailure("Modified Newton did not reach the residual tolerance.")

    return OptimizeResult(
        x=y * scale,
        fun=f.copy(),
        success=True,
        status=1,
        message=f"Newton/Broyden accepted after {total_iterations} iteration{'s' if total_iterations != 1 else ''}.",
        nfev=count,
        njev=builds,
        nit=total_iterations,
        broyden_updates=updates,
        jacobian=matrix.copy() if matrix is not None else None,
        jacobian_scale=scale.copy() if matrix is not None else None,
        jacobian_reused=reused_jacobian,
        jacobian_rebuilt=builds > 0,
        jacobian_secant_error=last_secant_error,
        cost=0.5 * float(np.dot(f, f)),
        optimality=float(np.max(np.abs(f))) if len(f) else 0.0,
        method="newton-broyden",
    )
