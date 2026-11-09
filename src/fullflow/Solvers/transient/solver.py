"""Public transient solver entry point."""

from __future__ import annotations

from typing import Any
import math
import time

from rich.console import Console

from fullflow.Solvers.steady_state.settings import LeastSquaresSettings, StateEvaluationSettings
from fullflow.Exceptions import SensorDataStop, TransientStepError
from fullflow.Exports.HDF5 import run_group_path

from .diagnostics import TransientPrinter
from .evaluation import TransientStateEvaluator
from .operations import StepDiagnostics, TransientStepSolve
from .results import TransientHistory, DiagnosticSummary
from .runtime import TransientRuntimeCache
from .settings import TransientSettings
from .events import SequenceSwitches


class Transient:
    """Backward-Euler transient solver for FullFlow networks."""

    def __init__(self, network) -> None:
        """Initialize the object and register any FullFlow state wiring."""
        self.network = network
        self.console = Console()
        self._runtime_cache: TransientRuntimeCache | None = None
        self._ignore_balances = None
        self._force_steady = None
        self._force_steady_exceptions = None
        self._linearization_source = None
        self.evaluator = TransientStateEvaluator(self._cache)
        self.step_solver = TransientStepSolve(
            network,
            self._cache,
            self.evaluator,
        )
        self.printer = TransientPrinter(network, self._cache, self.console)
        self.history = TransientHistory()
        self.step_diagnostics: list[StepDiagnostics] = []
        self.event_records: list[dict[str, Any]] = []
        self.event_probes = 0
        self.work_statistics = {}
        self._last_result = None
        self._run_metadata = None
        self._output_path = "transient"

    def linearize(
        self, inputs=None, outputs=None, filename=None, *, times=None,
        verbose=False,
    ):
        """Linearize the latest accepted point or selected saved times."""
        from fullflow.Solvers.linearization.solver import linearize
        return linearize(
            self, inputs=inputs, outputs=outputs, filename=filename, times=times,
            verbose=verbose,
        )

    def run(self, *args: Any, **kwargs: Any):
        """Alias for :meth:`solve` so ``Transient(network).run()`` works."""
        return self.solve(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        """Alias for :meth:`solve` so ``Transient(network)()`` works."""
        return self.solve(*args, **kwargs)

    def _refresh_runtime_cache(self) -> TransientRuntimeCache:
        """Force a fresh transient runtime view of the current network."""
        self._runtime_cache = TransientRuntimeCache(
            self.network,
            ignore_balances=self._ignore_balances,
            force_steady=self._force_steady,
            force_steady_exceptions=self._force_steady_exceptions,
        )
        return self._runtime_cache

    def _cache(self) -> TransientRuntimeCache:
        """Return a current runtime cache, refreshing if the network changed."""
        if self._runtime_cache is None:
            return self._refresh_runtime_cache()
        return self._runtime_cache.ensure_current()

    def _collect_sequence_breakpoints(self) -> tuple[float, ...]:
        """Return known tabular Sequence and command-trace times."""
        breakpoints: set[float] = set()

        for component in self.network.component_list:
            if getattr(component, "_table_sequence", False):
                times = getattr(component, "times", None)
                if times is not None:
                    try:
                        if hasattr(times, "is_assigned") and not times.is_assigned:
                            time_values = ()
                        else:
                            time_values = times.value if hasattr(times, "value") else times
                    except Exception:
                        time_values = ()

                    try:
                        for time_value in time_values:
                            breakpoint_time = float(time_value)
                            if math.isfinite(breakpoint_time):
                                breakpoints.add(breakpoint_time)
                    except (TypeError, ValueError):
                        pass

            command_breakpoints = getattr(type(component), "command_breakpoints", None)
            if callable(command_breakpoints):
                try:
                    for time_value in command_breakpoints(component):
                        breakpoint_time = float(time_value)
                        if math.isfinite(breakpoint_time):
                            breakpoints.add(breakpoint_time)
                except Exception:
                    continue

        return tuple(sorted(breakpoints))

    def _pick_timestep(
        self,
        dt: float,
        t_final: float,
        sequence_breakpoints: tuple[float, ...] = (),
    ) -> float:
        """Return the next timestep, shortened at final time or sequence breakpoints."""
        current_time = float(self.network.time.value)
        target_time = min(current_time + dt, t_final)

        tolerance = 1.0e-12 * max(1.0, abs(current_time), abs(target_time))

        for breakpoint_time in sequence_breakpoints:
            if breakpoint_time <= current_time + tolerance:
                continue

            if breakpoint_time > target_time + tolerance:
                break

            target_time = min(target_time, breakpoint_time, t_final)
            break

        return target_time - current_time

    @staticmethod
    def _validate_timestep(dt: float) -> float:
        """Return numeric ``dt`` and reject removed automatic timestep modes."""
        if isinstance(dt, str):
            raise ValueError(
                "dt must be a positive number. Automatic timestep modes "
                "were removed so timestep accuracy stays under user control."
            )

        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive. Got {dt}")

        return dt

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
        dt: float,
        t_final: float,
        filename: str | None = None,
        history: str = "full",
        verbose: bool = False,
        statistics: bool = False,
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
        ignore_balances=None,
        group: str | None = None,
    ):
        """Advance the network from its current time to ``t_final``."""
        return self._solve_once(
            dt=dt,
            t_final=t_final,
            filename=filename,
            group_path=run_group_path("transient", group=group),
            history=history,
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
            ignore_balances=ignore_balances,
            force_steady=None,
            force_steady_exceptions=None,
            metadata={"solve_type": "transient"},
        )

    def save(self, filename: str, group: str | None = None) -> None:
        """Save the latest completed solve without advancing the Network."""
        if self._last_result is None or self._linearization_source is None:
            raise RuntimeError("Complete a transient solve before saving results.")
        self._linearization_source.validate()
        path = self._output_path if group is None else run_group_path("transient", group=group)
        step_rows = self._diagnostic_rows(self.step_diagnostics) if self.history.mode == "full" else []
        self.history.save(
            filename, self.network, step_rows,
            group_path=path, metadata=self._run_metadata, result=self._last_result,
        )

    def _solve_once(
        self,
        dt: float,
        t_final: float,
        filename: str | None = None,
        history: str = "full",
        verbose: bool = False,
        statistics: bool = False,
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
        ignore_balances=None,
        force_steady=None,
        force_steady_exceptions=None,
        group_path: str = "transient",
        metadata: dict[str, Any] | None = None,
    ):
        """Advance the network from its current time to ``t_final``."""
        self._linearization_source = None
        self._last_result = None
        self._linearization_issue = None
        self._ignore_balances = ignore_balances
        self._force_steady = force_steady
        self._force_steady_exceptions = force_steady_exceptions
        prior_cache = self._runtime_cache
        reuse_allowed = (
            prior_cache is not None
            and prior_cache.version == self.network.version
            and prior_cache.ignore_balances == ignore_balances
            and prior_cache.force_steady == force_steady
            and prior_cache.force_steady_exceptions == force_steady_exceptions
        )

        dt = self._validate_timestep(dt)

        start_time_value = float(self.network.time.value)
        if not math.isfinite(start_time_value):
            raise ValueError("network.time must be finite.")
        if start_time_value > t_final:
            raise ValueError(
                f"network.time ({self.network.time.value}) is already greater than "
                f"t_final ({t_final})."
            )

        transient_settings = TransientSettings(
            dt=dt,
            t_final=t_final,
            max_step_retries=max_step_retries,
            minimum_dt=minimum_dt,
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

        self.history = TransientHistory(history)
        self.step_diagnostics = []
        summary_stats = DiagnosticSummary()
        if reuse_allowed and prior_cache.prepare_reused_run():
            cache = prior_cache
        else:
            cache = self._refresh_runtime_cache()

        run_metadata = dict(metadata or {})
        run_metadata.update(cache.dynamic_mode_summary())
        stop_reason = None

        solve_start_time = time.perf_counter()

        # Only event=True Sequences activate switching localization.
        switches = SequenceSwitches(self)
        switches.initialize()
        self.event_records = switches.records
        self.event_probes = 0
        self.work_statistics = {}

        # Initialize accepted state history before marching.
        self.step_solver.initialize(state_settings)
        self.network.reset_sensor_events()
        self.network.reset_sequence_aborts()
        self.network.initialize_sensor_conditions(float(self.network.time.value))
        self.history.append(self.network, float(self.network.time.value), cache)
        from fullflow.Solvers.linearization.checkpoints import discrete_signature
        previous_mode = discrete_signature(cache)
        sequence_breakpoints = self._collect_sequence_breakpoints()
        sampled = tuple(
            component for component in cache.component_list
            if getattr(component, "_sample_enabled", False)
        )
        for component in sampled:
            component.initialize_sampling(float(self.network.time.value))

        abort_record = None
        unfinished_endpoint = None
        events_in_interval = 0

        while float(self.network.time.value) < t_final:
            current_time = float(self.network.time.value)
            if unfinished_endpoint is None or current_time >= unfinished_endpoint - 1e-12 * max(1.0, abs(current_time)):
                unfinished_endpoint = None
                events_in_interval = 0
            pending_abort_time = self.network.next_sequence_abort_time(current_time)
            tolerance = 1.0e-12 * max(1.0, abs(current_time), abs(t_final))

            if pending_abort_time is not None and pending_abort_time <= current_time + tolerance:
                abort_record = self.network.check_sequence_abort(current_time)
                if abort_record is not None:
                    stop_reason = str(abort_record.get("message") or "Sequence abort requested.")
                    if verbose:
                        self.printer.print_sequence_abort(abort_record)
                break

            step_final = t_final
            if pending_abort_time is not None:
                step_final = min(step_final, float(pending_abort_time))

            if unfinished_endpoint is not None:
                step_final = min(step_final, unfinished_endpoint)
            # A sampled controller adds only known time boundaries, never a
            # nonlinear root search. The unsampled path remains unchanged.
            if sampled:
                for component in sampled:
                    next_time = component.next_sample_time()
                    if next_time is not None and next_time > current_time:
                        step_final = min(step_final, next_time)
            dt_step = self._pick_timestep(
                transient_settings.dt,
                step_final,
                sequence_breakpoints,
            )
            if dt_step <= 0.0:
                raise ValueError(f"dt is too small to advance network.time at t={current_time:g}.")

            trial_dt = dt_step
            retries = 0

            while True:
                t_new = min(current_time + trial_dt, t_final)
                tolerance = 1.0e-12 * max(1.0, abs(current_time), abs(t_final))
                if t_final - t_new <= tolerance:
                    t_new = t_final
                step_dt = t_new - current_time

                try:
                    if switches.sequences:
                        diagnostics, localized_event = switches.run_step(
                            t_new=t_new, dt=step_dt,
                            settings=least_squares_settings,
                            state_settings=state_settings,
                        )
                    else:
                        diagnostics = self.step_solver.run_once(
                            t_new=t_new, dt=step_dt,
                            least_squares_settings=least_squares_settings,
                            state_settings=state_settings,
                        )
                        localized_event = False
                    diagnostics.retries = retries
                    break

                except SensorDataStop as error:
                    stop_reason = str(error)
                    self.network.time.value = current_time
                    self._cache()
                    diagnostics = None
                    break

                except Exception as error:
                    if retries >= transient_settings.max_step_retries:
                        raise TransientStepError(
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
                        raise TransientStepError(
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

            if diagnostics is None:
                break

            summary_stats.append(diagnostics)
            if history == "full":
                self.step_diagnostics.append(diagnostics)
            else:
                self.step_diagnostics[:] = [diagnostics]
            self.event_probes = switches.probes
            if localized_event:
                events_in_interval += 1
                if events_in_interval > 32:
                    raise TransientStepError(
                        "More than 32 Sequence switches inside one requested timestep. "
                        "Use discrete controller functions with a real hysteresis band."
                    )
                if diagnostics.time < t_new - tolerance:
                    unfinished_endpoint = t_new
            elif unfinished_endpoint is not None and diagnostics.time >= unfinished_endpoint - tolerance:
                unfinished_endpoint = None
                events_in_interval = 0
            step_events = self.network.check_sensor_conditions(diagnostics.time)
            if step_events and verbose:
                self.printer.print_sensor_events(step_events)
            if step_events:
                self.network.activate_sequence_conditions(step_events)
                self.step_solver.invalidate_equation_scales()

            # All sampled controllers read the same accepted plant state,
            # then publish held commands and memory once per sample boundary.
            if sampled:
                pending_samples = [
                    (component, prepared)
                    for component in sampled
                    if (prepared := component.prepare_sample(diagnostics.time)) is not None
                ]
                for component, (command, memory) in pending_samples:
                    component.accept_sample(command, memory)

            abort_record = self.network.check_sequence_abort(diagnostics.time)
            if abort_record is not None:
                stop_reason = str(abort_record.get("message") or "Sequence abort requested.")
                if verbose:
                    self.printer.print_sequence_abort(abort_record)

            if statistics:
                self.printer.print_step(diagnostics)

            # Refresh structural changes before the next step.
            self._cache()
            current_mode = discrete_signature(self._cache())
            switched = current_mode != previous_mode
            if switched:
                self.step_solver.invalidate_equation_scales()
            previous_mode = current_mode
            issue = "An event or operating mode changed at this timestep." if localized_event or step_events or abort_record or switched else None
            self.history.append(self.network, diagnostics.time, self._cache(), issue)
            self._linearization_issue = issue

            if abort_record is not None:
                break

        elapsed_time = time.perf_counter() - solve_start_time
        self.work_statistics = dict(self.step_solver.work_statistics)
        self.work_statistics.update(
            event_probes=switches.probes,
            localized_switches=len(switches.records),
            reconciliation_residual_calls=switches.reconciliation_residual_calls,
            reconciliation_scipy_solves=switches.reconciliation_scipy_solves,
            reconciliation_reported_jacobian_builds=switches.reconciliation_reported_jacobian_builds,
            reconciliation_scale_jacobian_builds=switches.reconciliation_scale_jacobian_builds,
        )
        for key, value in self.work_statistics.items():
            run_metadata[f"solver_work_{key}"] = value
        if self.event_records:
            run_metadata["sequence_localized_event_count"] = len(self.event_records)
            run_metadata["sequence_localized_events"] = list(self.event_records)
        sensor_event_records = self.network.sensor_event_records()
        if sensor_event_records:
            run_metadata["sensor_event_count"] = len(sensor_event_records)
            for role in ("greenline", "blueline", "yellowline", "redline"):
                run_metadata[f"{role}_event_count"] = sum(
                    1 for row in sensor_event_records if row.get("role") == role
                )
        if abort_record is not None:
            run_metadata["sequence_aborted"] = True
            run_metadata["abort_sequence"] = abort_record.get("sequence", "")
            run_metadata["abort_condition"] = abort_record.get("condition", "")
            run_metadata["abort_time"] = abort_record.get("time", float(self.network.time.value))
            run_metadata["abort_trigger_time"] = abort_record.get("trigger_time", math.nan)
            run_metadata["abort_delay"] = abort_record.get("delay", 0.0)
            run_metadata["abort_message"] = abort_record.get("message", "")
        if stop_reason is not None:
            if abort_record is None:
                run_metadata["stopped_by_sensor"] = True
            run_metadata["stop_reason"] = stop_reason
        final_time = float(self.network.time.value)
        final_issue = self._linearization_issue
        self._linearization_source = self.history.capture_final(self.network, self._cache(), final_issue)
        stats = summary_stats.result(
            final_time=final_time, elapsed=elapsed_time,
            start_time=start_time_value, work=self.work_statistics,
            stop_reason=stop_reason,
        )
        result = self.history.result(
            self.network,
            metadata=run_metadata,
            solver=stats,
            events={
                "sensor": sensor_event_records,
                "sequence": list(self.event_records),
                "abort": None if abort_record is None else dict(abort_record),
            },
        )
        self._last_result = result
        self._run_metadata = run_metadata
        self._output_path = group_path
        if filename is not None:
            self.save(filename)

        if verbose:
            self.printer.print_summary(
                diagnostics_list=self.step_diagnostics,
                summary_statistics=summary_stats if history == "summary" else None,
                start_time=start_time_value,
                final_time=float(self.network.time.value),
                requested_dt=dt,
                method=least_squares_settings.solver_method,
                jac=least_squares_settings.jacobian_method,
                ftol=least_squares_settings.ftol,
                xtol=least_squares_settings.xtol,
                gtol=least_squares_settings.gtol,
                rtol=least_squares_settings.rtol,
                solve_time=elapsed_time,
                success=stop_reason is None,
                stop_reason=stop_reason,
                work_statistics=self.work_statistics,
            )
            self.printer.print_sensor_event_summary(self.network.sensor_event_list)
            self.printer.print_network_solution()

        return result
