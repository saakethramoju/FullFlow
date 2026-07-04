"""User-facing steady-state solver API.

This file is intentionally the only place most users need to understand. It
wraps the smaller implementation files in this package:

* ``runtime.py`` gathers and validates solver metadata from a ``Network``.
* ``evaluation.py`` settles component-derived states.
* ``operations.py`` performs one static evaluation or one nonlinear solve.
* ``models.py`` handles model-option selection and fallback.
* ``diagnostics.py`` prints optional Rich tables.

The split keeps the public API small while making numerical behavior, model
logic, and diagnostics independently debuggable.
"""

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
    """Solve or statically evaluate a FullFlow network.

    Parameters
    ----------
    network:
        The ``Network`` to evaluate. The network remains the source of truth for
        components, balances, models, and saved outputs. Solver-specific state is
        stored in this object and in a :class:`RuntimeCache`.

    Notes
    -----
    ``SteadyState`` is intentionally a thin orchestrator. It owns public method
    signatures, debug snapshots, and printing flags; lower-level files own the
    actual cache building, state propagation, model fallback, and least-squares
    call.
    """

    def __init__(self, network: Network) -> None:
        self.network = network
        self.console = Console()
        self.statistics = SolverStatistics(console=self.console)
        self._runtime_cache: RuntimeCache | None = None
        self._ignore_balances = None
        self._active_state_settings = StateEvaluationSettings()

        # Debug fields are populated during residual calls. They are useful when
        # SciPy raises from inside a network evaluation and you need to inspect
        # the most recent x/residual/error state from a notebook or debugger.
        self._last_debug_x: np.ndarray | None = None
        self._last_debug_residual: np.ndarray | None = None
        self._last_debug_error: Exception | None = None

        # Some thermodynamic backends legitimately fail for temporary nonlinear
        # trial points, for example an invalid pressure-enthalpy flash while
        # SciPy is estimating a Jacobian. Keep the last valid residual so those
        # trial points can be rejected without aborting the whole solve.
        #
        # Invalid-trial residuals are directional rather than a fixed wall such
        # as 1.0e12. A fixed wall makes every invalid point look the same to
        # SciPy, which can stall the trust-region search. The directional gain
        # below scales from the last valid residual and the normalized distance
        # from the last valid iteration vector.
        self._last_valid_x: np.ndarray | None = None
        self._last_valid_residual: np.ndarray | None = None
        self._invalid_residual_directional_gain = 10.0

        # Success metadata is populated by the one-shot operation helpers and
        # consumed by ``_print_last_success``.
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
        """Residual function passed to SciPy ``least_squares``.

        SciPy owns the iteration vector. This method writes that vector into the
        network's iteration States, recomputes derived states, then collects the
        component and balance residuals in the exact order expected by SciPy.
        """
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
        """Return a directional residual for invalid nonlinear trial points.

        External property packages can raise for temporary SciPy guesses that
        are outside their valid thermodynamic domain. When a previous valid
        residual exists, make the bad point look worse in the direction it moved
        from that valid point instead of returning a constant residual wall.

        A constant vector such as ``1.0e12`` gives SciPy almost no information:
        all invalid guesses have the same cost and nearly the same numerical
        derivative. This directional fallback preserves the sign pattern of the
        failed step and scales with the normalized distance from the last valid
        iteration vector.
        """
        if self._last_valid_residual is None:
            return None

        last_residual = np.array(self._last_valid_residual, dtype=float)
        count = len(last_residual)

        if count == 0:
            return np.array([], dtype=float)

        if self._last_valid_x is None or len(x) == 0:
            # Without a valid iteration-vector direction, fall back to the last
            # residual sign pattern. This path should only occur for unusual
            # setup cases after at least one valid residual was collected.
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
                # Map the iteration-vector direction into residual space. The
                # residual and iteration vectors can have different lengths, so
                # resize repeats/truncates the direction while keeping the bad
                # trial's sign information instead of replacing it with a wall.
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

        # Keep the invalid trial more expensive than the last valid point while
        # preserving the direction computed above. This prevents a lucky
        # cancellation from making an invalid property state look attractive.
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
        """Run a quasi-steady time sweep using the transient runtime machinery.

        Every dynamic component is forced to satisfy derivative = 0 at each
        timestep unless it is listed in ``exceptions``.  Results are written
        under the transient run path using the normal transient time-history
        HDF5 layout.
        """
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
        """
        Evaluate the network without nonlinear solving.

        This method repeatedly evaluates component equations until all derived
        ``State`` objects have converged, then exports the resulting network values.

        Unlike :meth:`solve`, no iteration variables are adjusted and no residuals
        are driven to zero. This is useful when:

        - All required states are already assigned.
        - The network contains no balances or iteration variables.
        - You want to inspect intermediate calculations.
        - You want to verify component equations before running a solve.

        Parameters
        ----------
        model : str, optional
            Name of a model option to evaluate. If omitted, the default model
            configuration is used.

        evaluate_all_model_options : bool, default=False
            If True, evaluates every available model option and returns the
            results for each one. Useful for comparing alternative component
            models without modifying the network.

        filename : str, optional
            Output HDF5 file path. ``.h5`` is added automatically when no
            extension is provided. ``.h5`` and ``.hdf5`` are the only supported
            export extensions.

            If provided, results are written to disk in addition to being returned.

        return_type : {"dict"}, default="dict"
            Format of the returned results. ``"dict"`` returns a list of
            exported records.

        verbose : bool, default=False
            Print model-selection and evaluation progress.

        state_max_passes : int, default=20
            Maximum number of passes used to settle derived states.

            Some states depend on other states that are computed later in the
            evaluation sequence. Multiple passes allow these dependencies to
            propagate through the network until convergence.

        state_tolerance : float, default=1e-10
            Convergence tolerance used when settling derived states.

            State evaluation stops when the largest state change between passes
            falls below this value.

        Returns
        -------
        dict or model result object
            Exported network records. Model-option evaluation returns a dict
            keyed by option name.

        Notes
        -----
        This method does not perform any nonlinear solving. Residual equations,
        balances, and iteration variables are ignored.
        """
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
        """
        Solve the network steady state.

        This method adjusts all iteration variables until the network residuals
        are minimized and all balance equations are satisfied.

        The typical solve sequence is:

        1. Select or build the requested model configuration.
        2. Run component ``pre_evaluation()`` hooks.
        3. Gather iteration variables and bounds.
        4. Solve the nonlinear system using SciPy ``least_squares``.
        5. Write the converged values back to the network.
        6. Re-evaluate derived states.
        7. Export and return the final network values.

        Parameters
        ----------
        model : str, optional
            Name of a model option to solve.

            If omitted, the default model configuration is used. If model
            fallback is enabled, alternative model options may be attempted
            automatically when the selected model fails.

        evaluate_all_model_options : bool, default=False
            Solve every available model option and return the results for each.

            This is useful for comparing multiple physical models, correlations,
            or component configurations.

        filename : str, optional
            Output HDF5 file path. ``.h5`` is added automatically when no
            extension is provided. ``.h5`` and ``.hdf5`` are the only supported
            export extensions.

            If provided, results are written to disk in addition to being returned.

        return_type : {"dict"}, default="dict"
            Format of the returned results.

        verbose : bool, default=False
            Print the final solver summary and final network solution.

        statistics : bool, default=False
            Print per-evaluation solver progress and export detailed statistics when filename is provided.

        static : bool, default=False
            Skip nonlinear solving and perform a static evaluation instead.

            Equivalent to calling :meth:`static_evaluate`.

        dt, t_final : float, optional
            When both are provided, ``SteadyState.solve`` performs a
            quasi-steady time sweep instead of a single steady-state solve.  The
            transient runtime is used to march from the current network time to
            ``t_final``, but every dynamic component is forced to satisfy
            derivative = 0 at each timestep unless it is listed in
            ``exceptions``.  HDF5 output is written under ``transient/runs``
            using the normal transient time-history layout.

        save_dt : float, optional
            Output interval for a quasi-steady time sweep.  If omitted, every
            accepted timestep is saved.

        exceptions : iterable of Component, optional
            Dynamic components that are excluded from the forced-steady
            treatment during a quasi-steady time sweep.  These components keep
            their normal transient integration residuals.

        max_step_retries : int, default=8
            Maximum automatic half-step retries for quasi-steady time sweeps.

        minimum_dt : float, optional
            Smallest automatic retry timestep for quasi-steady time sweeps.

        solver_method : str, default="trf"
            SciPy ``least_squares`` algorithm.

            Common options:

            - ``"trf"`` : Trust Region Reflective (recommended)
            - ``"dogbox"``
            - ``"lm"`` (unconstrained only)

        jacobian_method : str, default="3-point"
            Finite-difference Jacobian approximation method.

            Options:

            - ``"2-point"``
            - ``"3-point"``

        ftol : float, default=1e-8
            Relative cost-function convergence tolerance.

        xtol : float, default=1e-8
            Iteration-variable convergence tolerance.

        gtol : float, default=1e-8
            Gradient convergence tolerance.

        rtol : float, default=1e-2
            Residual acceptance tolerance.

            After convergence, the maximum absolute residual must be below this
            value for the solve to be considered successful.

        state_max_passes : int, default=5
            Maximum number of derived-state settling passes performed during
            solver iterations and after convergence.

        state_tolerance : float, default=1e-10
            Convergence tolerance used when settling derived states.

        ignore_balances : None, "all", or iterable of str, optional
            User ``Balance`` objects to exclude from this solve. Ignoring a
            balance removes both its residual and its associated iteration
            variable. Component balances are not affected.

        Returns
        -------
        dict or model result object
            Exported network records. Model-option evaluation returns a dict
            keyed by option name.

        Notes
        -----
        Solver variables are collected from component ``dynamics``, component
        ``balances``, and user ``Balance`` objects.  Bounds defined on those
        States are automatically passed to the nonlinear solver.

        For debugging a network, it is often useful to first call
        :meth:`static_evaluate` before attempting a full solve.
        """
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
