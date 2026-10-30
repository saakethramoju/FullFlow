"""Public transient solver entry point."""

from __future__ import annotations

from typing import Any
import math
import time

from rich.console import Console

from fullflow.Solvers.steady_state.models import ModelManager
from fullflow.Solvers.steady_state.settings import LeastSquaresSettings, StateEvaluationSettings
from fullflow.Exceptions import SensorDataStop, TransientStepError

from .diagnostics import TransientPrinter
from .evaluation import TransientStateEvaluator
from .operations import StepDiagnostics, TransientStepSolve
from .results import TransientHistory, format_records
from .runtime import TransientRuntimeCache
from .models import TransientModelOptionRunner
from .settings import TransientSettings


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
        self._linearization_points = {}
        self.evaluator = TransientStateEvaluator(self._cache)
        self.step_solver = TransientStepSolve(
            network,
            self._cache,
            self.evaluator,
        )
        self.printer = TransientPrinter(network, self._cache, self.console)
        self.history = TransientHistory()
        self.step_diagnostics: list[StepDiagnostics] = []

    def linearize(self, inputs=None, outputs=None, filename=None, *, times=None):
        """Linearize the latest accepted point or selected saved times."""
        from fullflow.Solvers.linearization.solver import linearize
        return linearize(self, inputs=inputs, outputs=outputs, filename=filename, times=times)

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
        next_save_time: float | None = None,
    ) -> float:
        """Return the next timestep, shortened at final, sequence, or save times."""
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

        if next_save_time is not None:
            if current_time + tolerance < next_save_time <= target_time + tolerance:
                target_time = min(target_time, next_save_time, t_final)

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
        if dt <= 0.0:
            raise ValueError(f"dt must be positive. Got {dt}")

        return dt

    @staticmethod
    def _validate_output_settings(save_dt: float | None) -> None:
        """Validate transient output-throttling settings."""
        if save_dt is not None and save_dt <= 0.0:
            raise ValueError(f"save_dt must be positive or None. Got {save_dt}")

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
        dt: float,
        t_final: float,
        model: str | Any | None = None,
        evaluate_all_model_options: bool = False,
        filename: str | None = None,
        return_type: str = "dict",
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
        save_dt: float | None = None,
        ignore_balances=None,
    ):
        """Advance the network with optional model-option selection."""

        self._linearization_source = None
        self._linearization_points = {}
        self._linearization_issue = None

        def run_once(
            filename: str | None = None,
            return_type: str = "dict",
            group_path: str = "transient/runs/base",
            metadata: dict[str, Any] | None = None,
            verbose_override: bool | None = None,
        ):
            return self._solve_no_model(
                dt=dt,
                t_final=t_final,
                filename=filename,
                return_type=return_type,
                verbose=verbose if verbose_override is None else bool(verbose_override),
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
                force_steady=None,
                force_steady_exceptions=None,
                group_path=group_path,
                metadata=metadata,
            )

        runner = TransientModelOptionRunner(
            self.network,
            ModelManager(self.network),
            self.printer,
        )
        return runner.run(
            model=model,
            evaluate_all_model_options=evaluate_all_model_options,
            filename=filename,
            return_type=return_type,
            verbose=verbose,
            run_once=run_once,
        )

    def _solve_no_model(
        self,
        dt: float,
        t_final: float,
        filename: str | None = None,
        return_type: str = "dict",
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
        save_dt: float | None = None,
        ignore_balances=None,
        force_steady=None,
        force_steady_exceptions=None,
        group_path: str = "transient/runs/base",
        metadata: dict[str, Any] | None = None,
    ):
        """Advance the network from its current time to ``t_final``."""
        self._linearization_source = None
        self._linearization_points = {}
        self._linearization_issue = None
        self._ignore_balances = ignore_balances
        self._force_steady = force_steady
        self._force_steady_exceptions = force_steady_exceptions
        self._runtime_cache = None

        self._validate_output_settings(save_dt)
        dt = self._validate_timestep(dt)

        start_time_value = float(self.network.time.value)
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

        self.history = TransientHistory()
        self.step_diagnostics = []
        cache = self._refresh_runtime_cache()

        run_metadata = dict(metadata or {})
        run_metadata.update(cache.dynamic_mode_summary())
        stop_reason = None

        solve_start_time = time.perf_counter()

        # Initialize accepted state history before marching.
        self.step_solver.initialize(state_settings)
        self.network.reset_sensor_events()
        self.network.reset_sequence_aborts()
        self.network.initialize_sensor_conditions(float(self.network.time.value))
        self.history.append(self.network, float(self.network.time.value))
        from fullflow.Solvers.linearization.checkpoints import OperatingCheckpoint, discrete_signature
        self._linearization_points[float(self.network.time.value)] = (OperatingCheckpoint(self.network, cache), None)
        previous_mode = discrete_signature(cache)
        sequence_breakpoints = self._collect_sequence_breakpoints()
        next_save_time = None if save_dt is None else float(self.network.time.value) + float(save_dt)

        abort_record = None

        while float(self.network.time.value) < t_final:
            current_time = float(self.network.time.value)
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

            dt_step = self._pick_timestep(
                transient_settings.dt,
                step_final,
                sequence_breakpoints,
                next_save_time,
            )
            if dt_step <= 0.0:
                break

            trial_dt = dt_step
            retries = 0

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

            self.step_diagnostics.append(diagnostics)
            save_output = self._should_save_output(
                time_value=diagnostics.time,
                t_final=t_final,
                next_save_time=next_save_time,
            )

            step_events = self.network.check_sensor_conditions(diagnostics.time)
            if step_events and verbose:
                self.printer.print_sensor_events(step_events)
            if step_events:
                self.network.activate_sequence_conditions(step_events)

            abort_record = self.network.check_sequence_abort(diagnostics.time)
            if abort_record is not None:
                stop_reason = str(abort_record.get("message") or "Sequence abort requested.")
                if verbose:
                    self.printer.print_sequence_abort(abort_record)

            if (step_events or abort_record is not None) and not save_output:
                self.history.append(self.network, diagnostics.time)
                save_output = True
            elif save_output:
                self.history.append(self.network, diagnostics.time)

            if save_dt is not None:
                tolerance = 1.0e-12 * max(1.0, abs(diagnostics.time), abs(t_final))
                while next_save_time is not None and next_save_time <= diagnostics.time + tolerance:
                    next_save_time += float(save_dt)

            if statistics:
                self.printer.print_step(diagnostics)

            # Refresh structural changes before the next step.
            self._cache()
            current_mode = discrete_signature(self._cache())
            switched = current_mode != previous_mode
            previous_mode = current_mode
            issue = "An event or operating mode changed at this timestep." if step_events or abort_record or switched else None
            retain = save_output and (save_dt is not None or diagnostics.time >= t_final - 1e-10)
            if retain or issue:
                checkpoint = OperatingCheckpoint(self.network, self._cache())
                self._linearization_points[float(diagnostics.time)] = (checkpoint, issue)
            self._linearization_issue = issue

            if abort_record is not None:
                break

        elapsed_time = time.perf_counter() - solve_start_time
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
        step_rows = self._diagnostic_rows(self.step_diagnostics)
        self.history.save(filename, self.network, step_rows, group_path=group_path, metadata=run_metadata)

        if verbose:
            self.printer.print_summary(
                diagnostics_list=self.step_diagnostics,
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
            )
            self.printer.print_sensor_event_summary(self.network.sensor_event_list)
            self.printer.print_network_solution()

        self._linearization_source = OperatingCheckpoint(self.network, self._cache())
        return format_records(self.history.public_records, return_type)
