"""Public transient solver entry point.

The solver implemented here is a fixed-step, implicit backward-Euler network
solver.  Components expose derivatives; the solver builds residuals.

For every component pair

    transient_variables[i]   -> x
    transient_derivatives[i] -> xdot

one timestep solves

    x_new - x_previous - dt * xdot_new = 0

where ``xdot_new`` is evaluated after all current SciPy guesses have been written
into the network and all components have run at ``network.time = t_new``.
"""

from __future__ import annotations

from typing import Any
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
    This implementation intentionally has no retry logic and no adaptive ODE
    error control.  Each timestep is a bounded least-squares solve.  If SciPy
    cannot satisfy the per-step residual tolerance with the current ``dt``, the
    solver raises an error.
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

    def _pick_timestep(self, dt: float, t_final: float) -> float:
        """Return the next fixed timestep without stepping past ``t_final``.

        Schedule-breakpoint snapping will be added with the schedule phase.  For
        now this simply trims the final step so the solution lands exactly on
        ``t_final``.
        """
        current_time = float(self.network.time.value)
        return min(dt, t_final - current_time)

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
                }
            )

        return rows

    def solve(
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
        gtol: float = 1e-12,
        rtol: float = 1e-10,
        state_max_passes: int = 5,
        state_tolerance: float = 1e-10,
    ):
        """Advance the network from its current time to ``t_final``.

        Parameters
        ----------
        dt : float
            Requested fixed timestep.  The final step is shortened if needed so
            the solver lands exactly on ``t_final``.

        t_final : float
            Final simulation time.  The starting time is the current value of
            ``network.time``.

        filename : str, optional
            Output HDF5 filename.  When provided, transient data is written to
            disk.  The file contains the accepted timestep history under
            ``/transient/history``, per-step solver data under
            ``/transient/steps``, and the final network state under
            ``/solution/final``.

        return_type : {"dict"}, default="dict"
            Return format.  ``"dict"`` returns a list of time-stamped history
            records.

        verbose : bool, default=False
            Print the final transient solver summary and the final network state.
            This mirrors the steady-state solver's verbose behavior.

        statistics : bool, default=False
            Print accepted-step progression as the simulation runs.  This is the
            option to use when you want lines like ``t=..., dt=..., residual=...``.

        solver_method : str, default="trf"
            SciPy ``least_squares`` method.  ``"trf"`` is recommended because it
            supports State bounds.  ``"lm"`` is allowed only for unbounded
            transient systems.

        jacobian_method : str, default="3-point"
            Finite-difference Jacobian approximation passed to SciPy.

        ftol, xtol, gtol : float
            SciPy least-squares convergence tolerances.

        rtol : float, default=1e-10
            Per-timestep residual acceptance tolerance.  After SciPy terminates,
            the recomputed final timestep residual must satisfy
            ``max(abs(residual)) <= rtol``.  Dynamic residuals are scaled by their
            state magnitude; algebraic residuals are used exactly as components
            and balances return them.

        state_max_passes : int, default=5
            Maximum number of repeated ``evaluate_states()`` passes inside each
            residual call.

        state_tolerance : float, default=1e-10
            Fixed-point convergence tolerance for derived-state evaluation.

        Returns
        -------
        list[dict]
            Time-stamped network export records for every accepted timestep,
            including the initial state.
        """
        transient_settings = TransientSettings(dt=dt, t_final=t_final)
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

        start_time_value = float(self.network.time.value)
        if start_time_value > t_final:
            raise ValueError(
                f"network.time ({self.network.time.value}) is already greater than "
                f"t_final ({t_final})."
            )

        solve_start_time = time.perf_counter()

        # Evaluate the initial condition and store transient history.  At this
        # point every transient State has ``state.previous == state.value``.
        self.step_solver.initialize(state_settings)
        self.history.append(self.network, float(self.network.time.value))

        while float(self.network.time.value) < t_final:
            dt_step = self._pick_timestep(transient_settings.dt, t_final)
            if dt_step <= 0.0:
                break

            t_new = float(self.network.time.value) + dt_step
            diagnostics = self.step_solver.run_once(
                t_new=t_new,
                dt=dt_step,
                least_squares_settings=least_squares_settings,
                state_settings=state_settings,
            )
            self.step_diagnostics.append(diagnostics)
            self.history.append(self.network, t_new)

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
                requested_dt=transient_settings.dt,
                method=least_squares_settings.solver_method,
                jac=least_squares_settings.jacobian_method,
                ftol=least_squares_settings.ftol,
                xtol=least_squares_settings.xtol,
                gtol=least_squares_settings.gtol,
                rtol=least_squares_settings.rtol,
                solve_time=elapsed_time,
            )
            self.printer.print_network_solution()

        return format_records(self.history.records, return_type)
