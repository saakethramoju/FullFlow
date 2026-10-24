"""User-facing steady-state solver API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console

from .diagnostics import SteadyStatePrinter
from .evaluation import StateEvaluator
from .models import ModelManager, ModelOptionRunner
from .operations import NonlinearSolve, StaticDiagnostics, StaticEvaluation, SolveDiagnostics
from .runtime import RuntimeCache
from .settings import LeastSquaresSettings, StateEvaluationSettings
from .statistics import SolverStatistics
from fullflow.Exceptions import SensorDataStop, SolverSetupError

if TYPE_CHECKING:
    from fullflow.System import Network


class SteadyState:
    """Steady-state, static-evaluation, model-option, and quasi-steady solver front end."""

    def __init__(self, network: Network) -> None:
        """Initialize the object and register any FullFlow state wiring."""
        self.network = network
        self.console = Console()
        self.statistics = SolverStatistics(console=self.console)
        self._runtime_cache: RuntimeCache | None = None
        self._ignore_balances = None
        self._active_state_settings = StateEvaluationSettings()

        # Retain the latest trial for failure diagnostics.
        self._last_debug_x: np.ndarray | None = None
        self._last_debug_residual: np.ndarray | None = None
        self._last_debug_error: Exception | None = None

        # Directional penalties let the solver reject invalid property states.
        self._last_valid_x: np.ndarray | None = None
        self._last_valid_residual: np.ndarray | None = None
        self._invalid_residual_directional_gain = 10.0

        # Operation helpers populate this success summary.
        self._last_success_kind: str | None = None
        self._last_static_elapsed_time: float | None = None
        self._last_solve_diagnostics: SolveDiagnostics | None = None

        self.evaluator = StateEvaluator(self._cache)
        self.model_manager = ModelManager(network)
        self.printer = SteadyStatePrinter(network, self._cache, self.console)
        self.static_runner = StaticEvaluation(network, self._refresh_runtime_cache, self.evaluator)
        self.nonlinear_runner = NonlinearSolve(
            network,
            self._refresh_runtime_cache,
            self.evaluator,
            self.residual,
            self.static_runner,
        )
        self.model_runner = ModelOptionRunner(
            network,
            self.model_manager,
            self.printer,
            self._print_last_success,
        )

    def run(self, *args: Any, **kwargs: Any):
        """Alias for :meth:`solve` so ``SteadyState(network).run()`` works."""
        return self.solve(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        """Alias for :meth:`solve` so ``SteadyState(network)()`` works."""
        return self.solve(*args, **kwargs)

    def _refresh_runtime_cache(self) -> RuntimeCache:
        """Force a fresh runtime view of the current network."""
        self._runtime_cache = RuntimeCache(
            self.network,
            ignore_balances=self._ignore_balances,
        )
        return self._runtime_cache

    def _cache(self) -> RuntimeCache:
        """Return a current runtime cache, refreshing if the network changed."""
        if self._runtime_cache is None:
            return self._refresh_runtime_cache()
        return self._runtime_cache.ensure_current()

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Residual function passed to SciPy ``least_squares``."""
        cache = self._cache()
        self._last_debug_x = np.array(x, dtype=float)
        self._last_debug_residual = None
        self._last_debug_error = None
        cache.assign_iteration_values(x)

        try:
            self.evaluate_network_states(
                max_passes=self._active_state_settings.max_passes,
                tolerance=self._active_state_settings.tolerance,
            )
            residual = cache.collect_residuals()
            self._last_debug_residual = residual
            self._last_valid_x = np.array(x, dtype=float)
            self._last_valid_residual = np.array(residual, dtype=float)
            self.statistics.record(x, residual, cache, phase="solver")
            return residual

        except Exception as error:
            if isinstance(error, SensorDataStop):
                raise

            self._last_debug_error = error
            penalty_residual = self._invalid_trial_residual(x, cache)

            if penalty_residual is not None:
                self._last_debug_residual = penalty_residual
                self.statistics.record(x, penalty_residual, cache, phase="solver")
                return penalty_residual

            try:
                self._last_debug_residual = cache.collect_residuals()
            except Exception:
                self._last_debug_residual = None

            raise SolverSetupError(
                "Solver encountered an error while evaluating the network "
                "inside evaluate_network_states().\n\n"
                f"Original error:\n{type(error).__name__}: {error}"
            ) from error

    def _invalid_trial_residual(self, x: np.ndarray, cache: RuntimeCache) -> np.ndarray | None:
        """Return a directional residual for invalid nonlinear trial points."""
        if self._last_valid_residual is None:
            return None

        last_residual = np.array(self._last_valid_residual, dtype=float)
        count = len(last_residual)

        if count == 0:
            return np.array([], dtype=float)

        if self._last_valid_x is None or len(x) == 0:
            # Fall back to the last valid residual direction.
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
                # Resize the trial direction to residual space.
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

        # Keep invalid trials more expensive than the last valid point.
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm == 0.0 or not np.isfinite(residual_norm):
            residual = residual_scale * direction
            residual_norm = float(np.linalg.norm(residual))

        if residual_norm < target_norm:
            residual *= target_norm / max(residual_norm, np.finfo(float).tiny)

        return residual

    def evaluate_network_states(
        self,
        max_passes: int = 20,
        tolerance: float = 1e-10,
    ) -> None:
        """Settle component-derived states without changing iteration variables."""
        self.evaluator.run(max_passes=max_passes, tolerance=tolerance)

    def _static_evaluate_once(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        state_settings: StateEvaluationSettings | None = None,
        group_path: str = "steady_state/runs/base",
        metadata: dict[str, Any] | None = None,
    ):
        """Run one static evaluation after model options have been selected."""
        solution, diagnostics = self.static_runner.run_once(
            filename=filename,
            return_type=return_type,
            state_settings=state_settings,
            group_path=group_path,
            metadata=metadata,
        )
        self._last_success_kind = "static"
        self._last_static_elapsed_time = diagnostics.elapsed_time
        self._last_solve_diagnostics = None
        return solution

    def _solve_once(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        least_squares_settings: LeastSquaresSettings | None = None,
        state_settings: StateEvaluationSettings | None = None,
        statistics: bool = False,
        statistics_filename: str | None = None,
        group_path: str = "steady_state/runs/base",
        metadata: dict[str, Any] | None = None,
    ):
        """Run one nonlinear solve after model options have been selected."""
        self._active_state_settings = state_settings or StateEvaluationSettings(max_passes=5)
        self.statistics.enabled = statistics
        solution, diagnostics = self.nonlinear_runner.run_once(
            filename=filename,
            return_type=return_type,
            least_squares_settings=least_squares_settings,
            state_settings=self._active_state_settings,
            statistics=self.statistics,
            statistics_filename=statistics_filename,
            group_path=group_path,
            metadata=metadata,
        )

        if isinstance(diagnostics, StaticDiagnostics):
            self._last_success_kind = "static"
            self._last_static_elapsed_time = diagnostics.elapsed_time
            self._last_solve_diagnostics = None
        else:
            self._last_success_kind = "solve"
            self._last_static_elapsed_time = None
            self._last_solve_diagnostics = diagnostics

        return solution

    def _solve_forced_steady_time_sweep_once(
        self,
        *,
        dt: float,
        t_final: float,
        filename: str | None = None,
        return_type: str = "dict",
        verbose: bool = False,
        statistics: bool = False,
        solver_method: str = "trf",
        jacobian_method: str = "3-point",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float | None = None,
        rtol: float = 1e-2,
        state_max_passes: int = 5,
        state_tolerance: float = 1e-10,
        max_step_retries: int = 8,
        minimum_dt: float | None = None,
        save_dt: float | None = None,
        ignore_balances=None,
        exceptions=None,
        group_path: str = "transient/runs/base",
        metadata: dict[str, Any] | None = None,
    ):
        """Run a quasi-steady time sweep using the transient runtime machinery."""
        from fullflow.Solvers.transient.solver import Transient

        metadata = {} if metadata is None else dict(metadata)
        metadata.setdefault("solve_type", "transient")
        metadata.setdefault("run_mode", "forced_steady_time_sweep")

        transient_solver = Transient(self.network)
        result = transient_solver._solve_no_model(
            dt=dt,
            t_final=t_final,
            filename=filename,
            return_type=return_type,
            verbose=verbose,
            statistics=statistics,
            solver_method=solver_method,
            jacobian_method=jacobian_method,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            rtol=rtol,
            state_max_passes=state_max_passes,
            state_tolerance=state_tolerance,
            max_step_retries=max_step_retries,
            minimum_dt=minimum_dt,
            save_dt=save_dt,
            ignore_balances=ignore_balances,
            force_steady="all",
            force_steady_exceptions=exceptions,
            group_path=group_path,
            metadata=metadata,
        )

        self._last_success_kind = "forced_steady_time_sweep"
        self._last_static_elapsed_time = None
        self._last_solve_diagnostics = None
        return result

    def static_evaluate(
        self,
        model: str | None = None,
        evaluate_all_model_options: bool = False,
        filename: str | None = None,
        return_type: str = "dict",
        verbose: bool = False,
        statistics: bool = False,
        state_max_passes: int = 20,
        state_tolerance: float = 1e-10,
    ):
        """Evaluate the network without nonlinear solving."""
        self._ignore_balances = None
        self._runtime_cache = None

        state_settings = StateEvaluationSettings(
            max_passes=state_max_passes,
            tolerance=state_tolerance,
        )

        def run_once(
            filename: str | None = None,
            return_type: str = "dict",
            statistics_filename: str | None = None,
            group_path: str = "steady_state/runs/base",
            metadata: dict[str, Any] | None = None,
        ):
            return self._static_evaluate_once(
                filename=filename,
                return_type=return_type,
                state_settings=state_settings,
                group_path=group_path,
                metadata=metadata,
            )

        return self.model_runner.run(
            model=model,
            evaluate_all_model_options=evaluate_all_model_options,
            filename=filename,
            return_type=return_type,
            verbose=verbose,
            statistics=statistics,
            run_once=run_once,
        )

    def solve(
        self,
        model: str | None = None,
        evaluate_all_model_options: bool = False,
        filename: str | None = None,
        return_type: str = "dict",
        verbose: bool = False,
        statistics: bool = False,
        static: bool = False,
        dt: float | None = None,
        t_final: float | None = None,
        save_dt: float | None = None,
        exceptions=None,
        max_step_retries: int = 8,
        minimum_dt: float | None = None,
        solver_method: str = "trf",
        jacobian_method: str = "3-point",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float | None = 1e-8,
        rtol: float = 1e-2,
        state_max_passes: int = 5,
        state_tolerance: float = 1e-10,
        ignore_balances=None,
    ):
        """Solve the network steady state."""
        self._ignore_balances = ignore_balances
        self._runtime_cache = None

        time_sweep_requested = dt is not None or t_final is not None
        if time_sweep_requested:
            if dt is None or t_final is None:
                raise ValueError("dt and t_final must be provided together for a SteadyState time sweep.")
            if static:
                raise ValueError("static=True cannot be combined with dt/t_final in SteadyState.solve().")

            def run_time_sweep_once(
                filename: str | None = None,
                return_type: str = "dict",
                statistics_filename: str | None = None,
                group_path: str = "transient/runs/base",
                metadata: dict[str, Any] | None = None,
            ):
                return self._solve_forced_steady_time_sweep_once(
                    dt=float(dt),
                    t_final=float(t_final),
                    filename=filename,
                    return_type=return_type,
                    verbose=verbose,
                    statistics=statistics,
                    solver_method=solver_method,
                    jacobian_method=jacobian_method,
                    ftol=ftol,
                    xtol=xtol,
                    gtol=gtol,
                    rtol=rtol,
                    state_max_passes=state_max_passes,
                    state_tolerance=state_tolerance,
                    max_step_retries=max_step_retries,
                    minimum_dt=minimum_dt,
                    save_dt=save_dt,
                    ignore_balances=ignore_balances,
                    exceptions=exceptions,
                    group_path=group_path,
                    metadata=metadata,
                )

            from fullflow.Solvers.transient.models import TransientModelOptionRunner

            runner = TransientModelOptionRunner(
                self.network,
                ModelManager(self.network),
                self.printer,
                solve_type="transient",
                metadata_solve_type="transient",
                metadata_extra={"run_mode": "forced_steady_time_sweep"},
            )
            def run_time_sweep_model_once(
                filename: str | None = None,
                return_type: str = "dict",
                group_path: str = "transient/runs/base",
                metadata: dict[str, Any] | None = None,
                verbose_override: bool | None = None,
            ):
                metadata = {} if metadata is None else dict(metadata)
                metadata.setdefault("run_mode", "forced_steady_time_sweep")
                return run_time_sweep_once(
                    filename=filename,
                    return_type=return_type,
                    group_path=group_path,
                    metadata=metadata,
                )

            return runner.run(
                model=model,
                evaluate_all_model_options=evaluate_all_model_options,
                filename=filename,
                return_type=return_type,
                verbose=verbose,
                run_once=run_time_sweep_model_once,
            )

        if exceptions is not None:
            raise ValueError("exceptions can only be used when dt and t_final are provided.")
        if save_dt is not None:
            raise ValueError("save_dt can only be used when dt and t_final are provided.")

        if static:
            return self.static_evaluate(
                model=model,
                evaluate_all_model_options=evaluate_all_model_options,
                filename=filename,
                return_type=return_type,
                verbose=verbose,
                statistics=statistics,
                state_max_passes=state_max_passes,
                state_tolerance=state_tolerance,
            )

        least_squares_settings = LeastSquaresSettings(
            solver_method=solver_method,
            jacobian_method=jacobian_method,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            rtol=rtol,
        )
        state_settings = StateEvaluationSettings(
            max_passes=state_max_passes,
            tolerance=state_tolerance,
        )

        def run_once(
            filename: str | None = None,
            return_type: str = "dict",
            statistics_filename: str | None = None,
            group_path: str = "steady_state/runs/base",
            metadata: dict[str, Any] | None = None,
        ):
            return self._solve_once(
                filename=filename,
                return_type=return_type,
                least_squares_settings=least_squares_settings,
                state_settings=state_settings,
                statistics=statistics,
                statistics_filename=statistics_filename,
                group_path=group_path,
                metadata=metadata,
            )

        return self.model_runner.run(
            model=model,
            evaluate_all_model_options=evaluate_all_model_options,
            filename=filename,
            return_type=return_type,
            verbose=verbose,
            statistics=statistics,
            run_once=run_once,
        )

    def _print_last_success(self, verbose: bool) -> None:
        """Print final summaries requested by the caller after a success."""
        if not verbose:
            return

        if self._last_success_kind == "static":
            self.printer.print_static(self._last_static_elapsed_time or 0.0)
        elif self._last_success_kind == "solve":
            diagnostics = self._last_solve_diagnostics
            if diagnostics is None:
                return
            settings = diagnostics.settings
            self.printer.print_solve(
                diagnostics.sol,
                diagnostics.x0,
                settings.solver_method,
                settings.jacobian_method,
                settings.ftol,
                settings.xtol,
                settings.gtol,
                settings.rtol,
                overconstrained=diagnostics.overconstrained,
                elapsed_time=diagnostics.elapsed_time,
            )

        self.printer.print_network_solution()
