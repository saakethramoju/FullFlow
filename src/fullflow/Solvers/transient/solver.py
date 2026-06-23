"""Public transient solver entry point.

The solver implemented here is a fixed-step, implicit backward-Euler network
solver.  Components expose derivatives; the solver builds residuals.

For every dynamic component triple

    transient_variables[i]   -> nonlinear solver unknown
    transient_states[i]      -> integrated/conserved quantity
    transient_derivatives[i] -> d(transient_states[i]) / dt

one timestep solves

    state_new - state_previous - dt * statedot_new = 0

where ``statedot_new`` is evaluated after all current SciPy guesses have been
written into the network and all components have run at
``network.time = t_new``.  By default ``transient_states`` is the same list as
``transient_variables``.
"""

from __future__ import annotations

from typing import Any
import math
import time

from rich.console import Console

from fullflow.Solvers.steady_state.settings import LeastSquaresSettings, StateEvaluationSettings

from .diagnostics import TransientPrinter
from .evaluation import TransientStateEvaluator
from .operations import StepDiagnostics, TransientStepSolve
from .results import TransientHistory, format_records
from .runtime import TransientRuntimeCache
from .settings import TransientSettings


class Transient:
    """Fixed-step implicit transient solver for a FullFlow ``Network``.

    Parameters
    ----------
    network : Network
        The network to advance in time.

    Notes
    -----
    Each timestep is a bounded least-squares solve.  If the requested timestep
    does not satisfy the per-step residual tolerance, the solver automatically
    retries smaller half-steps before raising an error.
    """

    def __init__(self, network) -> None:
        self.network = network
        self.console = Console()
        self._runtime_cache: TransientRuntimeCache | None = None
        self.evaluator = TransientStateEvaluator(self._cache)
        self.step_solver = TransientStepSolve(
            network,
            self._cache,
            self.evaluator,
        )
        self.printer = TransientPrinter(network, self._cache, self.console)
        self.history = TransientHistory()
        self.step_diagnostics: list[StepDiagnostics] = []

    def run(self, *args: Any, **kwargs: Any):
        """Alias for :meth:`solve` so ``Transient(network).run()`` works."""
        return self.solve(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        """Alias for :meth:`solve` so ``Transient(network)()`` works."""
        return self.solve(*args, **kwargs)

    def _refresh_runtime_cache(self) -> TransientRuntimeCache:
        """Force a fresh transient runtime view of the current network."""
        self._runtime_cache = TransientRuntimeCache(self.network)
        return self._runtime_cache

    def _cache(self) -> TransientRuntimeCache:
        """Return a current runtime cache, refreshing if the network changed."""
        if self._runtime_cache is None:
            return self._refresh_runtime_cache()
        return self._runtime_cache.ensure_current()

    def _collect_schedule_breakpoints(self) -> tuple[float, ...]:
        """Return all known tabular Schedule times in the current network."""
        breakpoints: set[float] = set()

        for component in self.network.component_list:
            if not getattr(component, "_table_schedule", False):
                continue

            times = getattr(component, "times", None)
            if times is None:
                continue

            try:
                if hasattr(times, "is_assigned") and not times.is_assigned:
                    continue
                time_values = times.value if hasattr(times, "value") else times
            except Exception:
                continue

            try:
                for time_value in time_values:
                    breakpoint_time = float(time_value)
                    if math.isfinite(breakpoint_time):
                        breakpoints.add(breakpoint_time)
            except (TypeError, ValueError):
                continue

        return tuple(sorted(breakpoints))

    def _pick_timestep(
        self,
        dt: float,
        t_final: float,
        schedule_breakpoints: tuple[float, ...] = (),
        next_save_time: float | None = None,
    ) -> float:
        """Return the next timestep, shortened at final, schedule, or save times."""
        current_time = float(self.network.time.value)
        target_time = min(current_time + dt, t_final)

        tolerance = 1.0e-12 * max(1.0, abs(current_time), abs(target_time))

        for breakpoint_time in schedule_breakpoints:
            if breakpoint_time <= current_time + tolerance:
                continue

            if breakpoint_time > target_time + tolerance:
                break

            target_time = min(target_time, breakpoint_time, t_final)
            break

        if next_save_time is not None:
            if current_time + tolerance < next_save_time <= target_time + tolerance:
                target_time = min(target_time, next_save_time, t_final)

        return target_time - current_time

    @staticmethod
    def _validate_output_settings(save_dt: float | None) -> None:
        """Validate transient output-throttling settings."""
        if save_dt is not None and save_dt <= 0.0:
            raise ValueError(f"save_dt must be positive or None. Got {save_dt}")

    @staticmethod
    def _auto_timestep(
        *,
        start_time: float,
        t_final: float,
        save_dt: float | None = None,
    ) -> float:
        """Choose a simple starting timestep for ``dt="auto"``."""
        span = float(t_final) - float(start_time)

        if span <= 0.0:
            return 1.0

        dt = span / 100.0

        if save_dt is not None:
            dt = min(dt, float(save_dt))

        return max(dt, span * 1.0e-12)

    @staticmethod
    def _adaptive_ceiling(
        *,
        start_dt: float,
        start_time: float,
        t_final: float,
    ) -> float:
        """Return the private maximum timestep used by adaptive mode."""
        span = max(float(t_final) - float(start_time), float(start_dt))
        return max(float(start_dt), span / 20.0)

    @staticmethod
    def _step_function_evaluations(diagnostics: StepDiagnostics) -> int:
        """Return the SciPy function evaluation count for one accepted step."""
        sol = diagnostics.sol
        if sol is None:
            return 0

        return int(getattr(sol, "nfev", 0) or 0)

    @classmethod
    def _adapt_timestep(
        cls,
        *,
        current_dt: float,
        picked_dt: float,
        diagnostics: StepDiagnostics,
        settings: TransientSettings,
        rtol: float,
    ) -> float:
        """Choose the next timestep for simple adaptive mode."""
        floor = settings.retry_floor
        ceiling = settings.timestep_ceiling
        next_dt = float(current_dt)

        picked_limited = picked_dt < 0.999999 * current_dt
        nfev = cls._step_function_evaluations(diagnostics)

        if diagnostics.retries:
            next_dt = diagnostics.dt
        elif not picked_limited:
            if diagnostics.max_residual <= 0.01 * rtol and nfev <= 6:
                next_dt = 1.25 * current_dt
            elif diagnostics.max_residual >= 0.5 * rtol or nfev >= 20:
                next_dt = 0.75 * current_dt

        return min(max(next_dt, floor), ceiling)

    @staticmethod
    def _resolve_timestep_input(
        *,
        dt: float | str,
        adaptive: bool,
        start_time: float,
        t_final: float,
        save_dt: float | None,
    ) -> tuple[float, bool]:
        """Return ``(initial_dt, adaptive)`` from the public timestep inputs."""
        if isinstance(dt, str):
            if dt.lower() != "auto":
                raise ValueError("dt must be a positive number or 'auto'.")

            return (
                Transient._auto_timestep(
                    start_time=start_time,
                    t_final=t_final,
                    save_dt=save_dt,
                ),
                True,
            )

        return float(dt), bool(adaptive)

    @staticmethod
    def _should_save_output(
        *,
        time_value: float,
        t_final: float,
        next_save_time: float | None,
    ) -> bool:
        """Return True when an accepted timestep should be written to history."""
        tolerance = 1.0e-12 * max(1.0, abs(time_value), abs(t_final))

        if time_value >= t_final - tolerance:
            return True

        if next_save_time is None:
            return True

        return time_value >= next_save_time - tolerance

    @staticmethod
    def _diagnostic_rows(diagnostics_list: list[StepDiagnostics]) -> list[dict[str, Any]]:
        """Convert accepted-step diagnostics to HDF5-friendly row dictionaries."""
        rows: list[dict[str, Any]] = []

        for index, diagnostics in enumerate(diagnostics_list, start=1):
            sol = diagnostics.sol
            rows.append(
                {
                    "step": index,
                    "time": diagnostics.time,
                    "dt": diagnostics.dt,
                    "max_abs_residual": diagnostics.max_residual,
                    "rms_residual": diagnostics.rms_residual,
                    "solve_time_s": diagnostics.elapsed_time,
                    "success": True if sol is None else bool(getattr(sol, "success", False)),
                    "status": None if sol is None else getattr(sol, "status", None),
                    "message": "No nonlinear solve" if sol is None else str(getattr(sol, "message", "")),
                    "function_evaluations": 0 if sol is None else getattr(sol, "nfev", None),
                    "jacobian_evaluations": 0 if sol is None else getattr(sol, "njev", None),
                    "cost": 0.0 if sol is None else getattr(sol, "cost", None),
                    "optimality": 0.0 if sol is None else getattr(sol, "optimality", None),
                    "variable_count": len(diagnostics.x0),
                    "residual_count": len(diagnostics.residual),
                    "retries": diagnostics.retries,
                }
            )

        return rows

    def solve(
        self,
        dt: float | str,
        t_final: float,
        filename: str | None = None,
        return_type: str = "dict",
        verbose: bool = False,
        statistics: bool = False,
        adaptive: bool = False,
        solver_method: str = "trf",
        jacobian_method: str = "3-point",
        ftol: float = 1e-12,
        xtol: float = 1e-12,
        gtol: float | None = None,
        rtol: float = 1e-8,
        state_max_passes: int = 5,
        state_tolerance: float = 1e-10,
        max_step_retries: int = 8,
        minimum_dt: float | None = None,
        save_dt: float | None = None,
    ):
        """Advance the network from its current time to ``t_final``.

        Parameters
        ----------
        dt : float or "auto"
            Timestep control. A number gives a fixed timestep by default. A
            number with ``adaptive=True`` gives the starting timestep for simple
            automatic timestep adjustment. ``dt="auto"`` lets FullFlow choose a
            starting timestep and automatically adjust it. Steps are still
            shortened when needed to land exactly on ``t_final``, saved output
            times, and tabular ``Schedule`` time points.

        t_final : float
            Final simulation time.  The starting time is the current value of
            ``network.time``.

        filename : str, optional
            Output HDF5 filename.  When provided, transient data is written to
            disk.  The file contains saved transient history under
            ``/<network>/transient`` and the final network state under
            ``/<network>/transient/final``.

        save_dt : float, optional
            Output timestep. If omitted, every accepted solver step is saved. If
            provided, the solver still advances with its normal internal
            timestep, but output is saved only at multiples of ``save_dt`` and at
            final time. Timesteps are shortened when needed so saved output lands
            exactly on the requested output times.

        return_type : {"dict"}, default="dict"
            Return format.  ``"dict"`` returns a list of time-stamped tracked
            records.

        verbose : bool, default=False
            Print the final transient solver summary and the final network state.
            This mirrors the steady-state solver's verbose behavior.

        statistics : bool, default=False
            Print accepted-step progression as the simulation runs.  This is the
            option to use when you want lines like ``t=..., dt=..., residual=...``.

        adaptive : bool, default=False
            If ``False``, numeric ``dt`` is used as a fixed timestep with only
            failed-step retries. If ``True``, numeric ``dt`` is used as the
            starting timestep and FullFlow grows or shrinks later timesteps
            based on solve difficulty. ``dt="auto"`` turns this on automatically.

        solver_method : str, default="trf"
            SciPy ``least_squares`` method.  ``"trf"`` is recommended because it
            supports State bounds.  ``"lm"`` is allowed only for unbounded
            transient systems.

        jacobian_method : str, default="3-point"
            Finite-difference Jacobian approximation passed to SciPy.

        ftol, xtol : float
            SciPy least-squares convergence tolerances.

        gtol : float or None, default=None
            SciPy gradient convergence tolerance. ``None`` disables gradient-based
            termination, which avoids false convergence for small normalized
            transient residuals with weak finite-difference gradients.

        rtol : float, default=1e-8
            Per-timestep residual acceptance tolerance.  After SciPy terminates,
            the recomputed final timestep residual must satisfy
            ``max(abs(residual)) <= rtol``.  Internally generated integration
            residuals are normalized by the state/change scale; algebraic
            residuals are used exactly as components and balances return them.
            SciPy still uses the stricter ``ftol`` and ``xtol`` above.
            By default ``gtol`` is disabled for transient solves because small
            residual magnitudes can make gradient-based termination stop before
            the timestep residual is actually accepted.

        state_max_passes : int, default=5
            Maximum number of repeated ``evaluate_states()`` passes inside each
            residual call.

        state_tolerance : float, default=1e-10
            Fixed-point convergence tolerance for derived-state evaluation.

        max_step_retries : int, default=8
            Number of automatic half-step retries allowed when a timestep does
            not satisfy the acceptance residual.

        minimum_dt : float, optional
            Smallest automatic retry timestep.  If omitted, the retry floor is
            ``dt * 1e-9``.

        save_dt
            Output-throttling control. It does not change the model equations;
            it only controls which accepted timesteps are stored in memory and
            written to HDF5.

        Returns
        -------
        list[dict]
            Time-stamped tracked records for every accepted timestep, including
            the evaluated initial state.
        """
        self._validate_output_settings(save_dt)

        start_time_value = float(self.network.time.value)
        if start_time_value > t_final:
            raise ValueError(
                f"network.time ({self.network.time.value}) is already greater than "
                f"t_final ({t_final})."
            )

        initial_dt, adaptive = self._resolve_timestep_input(
            dt=dt,
            adaptive=adaptive,
            start_time=start_time_value,
            t_final=t_final,
            save_dt=save_dt,
        )

        maximum_dt = self._adaptive_ceiling(
            start_dt=initial_dt,
            start_time=start_time_value,
            t_final=t_final,
        ) if adaptive else initial_dt

        transient_settings = TransientSettings(
            dt=initial_dt,
            t_final=t_final,
            adaptive=adaptive,
            max_step_retries=max_step_retries,
            minimum_dt=minimum_dt,
            maximum_dt=maximum_dt,
        )
        transient_settings.validate()

        least_squares_settings = LeastSquaresSettings(
            solver_method=solver_method,
            jacobian_method=jacobian_method,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            rtol=rtol,
        )
        least_squares_settings.validate()

        state_settings = StateEvaluationSettings(
            max_passes=state_max_passes,
            tolerance=state_tolerance,
        )
        state_settings.validate()

        self.history = TransientHistory()
        self.step_diagnostics = []
        self._refresh_runtime_cache()

        solve_start_time = time.perf_counter()

        # Evaluate the initial condition and store transient history.  At this
        # point every transient State has ``state.previous == state.value``.
        self.step_solver.initialize(state_settings)
        self.history.append(self.network, float(self.network.time.value))
        schedule_breakpoints = self._collect_schedule_breakpoints()
        next_save_time = None if save_dt is None else float(self.network.time.value) + float(save_dt)
        active_dt = transient_settings.dt

        while float(self.network.time.value) < t_final:
            dt_step = self._pick_timestep(
                active_dt,
                t_final,
                schedule_breakpoints,
                next_save_time,
            )
            if dt_step <= 0.0:
                break

            current_time = float(self.network.time.value)
            trial_dt = dt_step
            retries = 0
            last_error: Exception | None = None

            while True:
                t_new = min(current_time + trial_dt, t_final)
                tolerance = 1.0e-12 * max(1.0, abs(current_time), abs(t_final))
                if t_final - t_new <= tolerance:
                    t_new = t_final
                step_dt = t_new - current_time

                try:
                    diagnostics = self.step_solver.run_once(
                        t_new=t_new,
                        dt=step_dt,
                        least_squares_settings=least_squares_settings,
                        state_settings=state_settings,
                    )
                    diagnostics.retries = retries
                    break

                except Exception as error:
                    last_error = error

                    if retries >= transient_settings.max_step_retries:
                        raise RuntimeError(
                            "Transient timestep failed after automatic retry.\n"
                            f"time = {current_time:.9g}\n"
                            f"requested dt = {dt_step:.9g}\n"
                            f"last attempted dt = {trial_dt:.9g}\n"
                            f"retries = {retries}\n"
                            f"minimum dt = {transient_settings.retry_floor:.9g}\n"
                            "Last error:\n"
                            f"{error}"
                        ) from error

                    next_dt = 0.5 * trial_dt
                    if next_dt < transient_settings.retry_floor:
                        raise RuntimeError(
                            "Transient timestep failed before reaching an acceptable residual, "
                            "and the next retry would be below minimum_dt.\n"
                            f"time = {current_time:.9g}\n"
                            f"requested dt = {dt_step:.9g}\n"
                            f"last attempted dt = {trial_dt:.9g}\n"
                            f"next attempted dt = {next_dt:.9g}\n"
                            f"minimum dt = {transient_settings.retry_floor:.9g}\n"
                            "Last error:\n"
                            f"{error}"
                        ) from error

                    retries += 1
                    trial_dt = next_dt
                    self.network.time.value = current_time
                    self._cache()

            self.step_diagnostics.append(diagnostics)
            save_output = self._should_save_output(
                time_value=diagnostics.time,
                t_final=t_final,
                next_save_time=next_save_time,
            )

            if save_output:
                self.history.append(self.network, diagnostics.time)

            if save_dt is not None:
                tolerance = 1.0e-12 * max(1.0, abs(diagnostics.time), abs(t_final))
                while next_save_time is not None and next_save_time <= diagnostics.time + tolerance:
                    next_save_time += float(save_dt)

            if transient_settings.adaptive:
                active_dt = self._adapt_timestep(
                    current_dt=active_dt,
                    picked_dt=dt_step,
                    diagnostics=diagnostics,
                    settings=transient_settings,
                    rtol=least_squares_settings.rtol,
                )

            if statistics:
                self.printer.print_step(diagnostics)

            # The runtime cache may become stale if a component modified network
            # structure during the step.  Refresh lazily for the next step.
            self._cache()

        elapsed_time = time.perf_counter() - solve_start_time
        step_rows = self._diagnostic_rows(self.step_diagnostics)
        self.history.save(filename, self.network, step_rows)

        if verbose:
            self.printer.print_summary(
                diagnostics_list=self.step_diagnostics,
                start_time=start_time_value,
                final_time=float(self.network.time.value),
                requested_dt=dt,
                adaptive=transient_settings.adaptive,
                method=least_squares_settings.solver_method,
                jac=least_squares_settings.jacobian_method,
                ftol=least_squares_settings.ftol,
                xtol=least_squares_settings.xtol,
                gtol=least_squares_settings.gtol,
                rtol=least_squares_settings.rtol,
                solve_time=elapsed_time,
            )
            self.printer.print_network_solution()

        return format_records(self.history.public_records, return_type)
