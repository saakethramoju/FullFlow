# SteadyState.py
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares, Bounds, root
from rich.console import Console
from rich.table import Table
from rich import box
import time

from fullflow.System.Composition import Composition
from .runtime import RuntimeCache

if TYPE_CHECKING:
    from fullflow.System import Network


class SteadyState:
    """
    Steady-state nonlinear network solver.

    `SteadyState` solves a network of algebraic equations by adjusting
    iteration variables until all component and balance residuals are driven
    to zero.

    The solver automatically propagates derived states, evaluates component
    models, assembles the residual vector, and calls
    `scipy.optimize.least_squares()` to compute a converged solution.

    Parameters
    ----------
    network : Network
        Network to evaluate and solve

    Notes
    -----
    The nonlinear system is written as:

        ``F(x) = 0``

    where:

        ``x``
            Vector of iteration variables

        ``F``
            Vector of component and balance residuals

    Iteration variables may include:

    * pressure
    * temperature
    * enthalpy
    * mass flow
    * friction factor
    * shaft speed
    * Balance variables

    Residual equations may include:

    * mass conservation
    * energy conservation
    * momentum equations
    * constitutive correlations
    * user-defined component residuals
    * Balance residuals

    Before residuals are evaluated, the solver repeatedly evaluates
    component states until derived quantities settle. This allows
    component ordering to be largely independent of network behavior.

    Example
    -------
    Create and solve a network:

        ``solver = SteadyState(network)``

        ``solver.solve()``

    Solve with verbose diagnostics:

        ``solver.solve(verbose=True)``

    Export results:

        ``solver.solve(filename="solution.xlsx")``

    Evaluate without nonlinear solving:

        ``solver.static_evaluate()``

    Model Evaluation
    ----------------
    `SteadyState` integrates directly with `Model` objects.

    A specific model can be solved:

        ``solver.solve(model="NozzleModel")``

    Every model option can be evaluated automatically:

        ``solver.solve(``
        ``    model="NozzleModel",``
        ``    evaluate_all_model_options=True``
        ``)``

    Failed model options are skipped and reported.

    State Propagation
    -----------------
    Most network quantities are derived rather than solved directly.

    Examples include:

    * density
    * viscosity
    * Reynolds number
    * fluid properties
    * heat-transfer coefficients
    * species composition

    During each residual evaluation:

        ``assign iteration variables``
        ``→ evaluate components``
        ``→ propagate derived states``
        ``→ repeat until settled``
        ``→ evaluate residuals``

    Iteration variables are protected during state propagation so that
    component evaluations cannot overwrite the solver's current iterate.

    Features
    --------
    * Nonlinear steady-state solving
    * Static network evaluation
    * Automatic state propagation
    * Balance support
    * Overdetermined system support
    * Model option switching
    * Solution export to CSV, JSON, and Excel
    * Rich terminal diagnostics
    * Verbose solver reporting

    See Also
    --------
    `Network`
        Container that owns components and balances

    `Balance`
        Additional solve equation and iteration variable

    `Model`
        Alternative component implementations

    `Component`
        Base class for all network elements
    """

    def __init__(self, network: Network):
        self.network = network
        self.console = Console()

        # Debug snapshots used when scipy fails before returning a solution.
        # These let verbose=True still print the last attempted variables,
        # variable adjustments, and any residuals that were available.
        self._last_debug_x = None
        self._last_debug_residual = None
        self._last_debug_error = None

        # Default state-settling controls used inside residual(), since scipy
        # calls residual(x) without passing solve() keyword arguments.
        self._state_max_passes = 20
        self._state_tolerance = 1e-10

        # Fast per-solve cache. Rebuilt at the beginning of each solve/static
        # evaluation and after model replacement.
        self._runtime_cache: RuntimeCache | None = None

    def run(self, *args, **kwargs):
        """Alias for solve(), useful when treating a solver like a component."""
        return self.solve(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call solve() directly with SteadyState(network)(...)."""
        return self.solve(*args, **kwargs)

    # ------------------------------------------------------------------
    # Residual function passed to scipy
    # ------------------------------------------------------------------

    def _refresh_runtime_cache(self) -> RuntimeCache:
        """Rebuild and return the fast per-solve runtime cache."""
        self._runtime_cache = RuntimeCache(self.network)
        return self._runtime_cache

    def _cache(self) -> RuntimeCache:
        """Return a current runtime cache, rebuilding only if the network changed."""
        if self._runtime_cache is None:
            return self._refresh_runtime_cache()
        return self._runtime_cache.ensure_current()

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        Map solver vector x to residual vector r.

        scipy calls this many times while solving, so this path uses cached
        network references instead of repeatedly scanning the full network.
        """
        cache = self._cache()

        # Save the current attempted iterate for failure diagnostics.
        self._last_debug_x = np.array(x, dtype=float)
        self._last_debug_residual = None
        self._last_debug_error = None

        # Assign current solver iterate to network State-like objects.
        cache.assign_iteration_values(x)

        try:
            # Propagate explicit component states to a consistent state.
            self.evaluate_network_states(
                max_passes=self._state_max_passes,
                tolerance=self._state_tolerance,
            )

            # Collect component and balance residuals.
            r = cache.collect_residuals()
            self._last_debug_residual = r

            return r

        except Exception as e:
            # Preserve the original error and whatever residuals can still
            # be collected, then re-raise so failures are not masked.
            self._last_debug_error = e

            try:
                self._last_debug_residual = cache.collect_residuals()
            except Exception:
                self._last_debug_residual = None

            raise RuntimeError(
                "Solver encountered an error while evaluating the network "
                "inside evaluate_network_states().\n\n"
                f"Original error:\n{type(e).__name__}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Order-independent state evaluation for steady-state solving
    # ------------------------------------------------------------------

    def evaluate_network_states(
        self,
        max_passes: int = 20,
        tolerance: float = 1e-10,
    ) -> None:
        """
        Repeatedly evaluate components until derived State values settle.

        Iteration variables are protected because they belong to the nonlinear
        solver, not to explicit component propagation. The implementation uses
        a cached list of State-like references and evaluate_states callables so
        residual evaluations avoid repeated full-network introspection.
        """
        cache = self._cache()

        # Save current solver-owned variables.
        iteration_snapshot = cache.snapshot_iteration_variables()

        for _ in range(max_passes):
            # Track non-iteration states before this pass.
            old_values = cache.collect_state_values()

            for evaluate_states in cache.evaluate_state_callables:
                # Restore solver variables before each component.
                cache.restore_iteration_variables(iteration_snapshot)

                # Let the component update its derived states.
                evaluate_states()

                # Restore solver variables again in case the component changed them.
                cache.restore_iteration_variables(iteration_snapshot)

            # Track non-iteration states after this pass.
            new_values = cache.collect_state_values()

            # Stop when another pass would not materially change the network.
            if cache.max_state_change(old_values, new_values) < tolerance:
                return

    def _snapshot_iteration_variables(self) -> dict[int, float]:
        """Backward-compatible helper for external/debug use."""
        return {
            id(var): float(var.value)
            for var in self._cache().iteration_variables
            if var.is_assigned
        }

    def _restore_iteration_variables(
        self,
        snapshot: dict[int, float],
    ) -> None:
        """Backward-compatible helper for external/debug use."""
        for var in self._cache().iteration_variables:
            key = id(var)
            if key in snapshot:
                var.value = snapshot[key]

    def _collect_state_values(self) -> dict[int, float]:
        """Backward-compatible helper for external/debug use."""
        return self._cache().collect_state_values()

    def _max_state_change(
        self,
        old: dict[int, float],
        new: dict[int, float],
    ) -> float:
        """Backward-compatible helper for external/debug use."""
        return RuntimeCache.max_state_change(old, new)


    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _get_model(self, model):
        """
        Return a Model object from either a Model instance or a model name.
        """
        if model is None:
            return None

        if hasattr(model, "build") and hasattr(model, "available_options"):
            return model

        for candidate in self.network.model_list:
            if candidate.name == model:
                return candidate

        raise ValueError(
            f"Unknown model {model!r}. "
            f"Available models are {[m.name for m in self.network.model_list]}."
        )

    def _build_unbuilt_models(self, exclude_model=None) -> None:
        """
        Build the first option for every unbuilt registered Model.

        If exclude_model is provided, that Model is skipped because the caller
        is explicitly controlling its active option.
        """
        for model in self.network.model_list:
            if model is exclude_model:
                continue

            if model.active_component is None:
                model.build()

    def _active_model_summary(self) -> str:
        """
        Return a compact summary of currently active model choices.
        """
        if not self.network.model_list:
            return "<none>"

        choices = []

        for model in self.network.model_list:
            option = model.active_option_name or "<unbuilt>"
            choices.append(f"{model.name}={option}")

        return ", ".join(choices)

    def _record_model_failure(
        self,
        failures: list[dict],
        error: Exception,
        model=None,
    ) -> None:
        """
        Store a concise model-option failure record for later printing.
        """
        failures.append({
            "model": model.name if model is not None else "<network>",
            "option": (
                model.active_option_name
                if model is not None
                else self._active_model_summary()
            ),
            "error_type": type(error).__name__,
            "error": str(error).splitlines()[0],
        })

    def _print_model_failures(self, failures: list[dict]) -> None:
        """
        Print model options that failed and were skipped.
        """
        if not failures:
            return

        table = Table(
            title="Skipped Model Options",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )

        table.add_column("Model", style="#D84135", no_wrap=True)
        table.add_column("Option", style="#fdf0d5", no_wrap=True)
        table.add_column("Error Type", style="#3B629E", no_wrap=True)
        table.add_column("Error")

        for failure in failures:
            table.add_row(
                str(failure["model"]),
                str(failure["option"]),
                str(failure["error_type"]),
                str(failure["error"]),
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _safe_sheet_name(self, name: str) -> str:
        """
        Return an Excel-safe worksheet name.
        """
        invalid_chars = ["\\", "/", "*", "[", "]", ":", "?"]

        for char in invalid_chars:
            name = name.replace(char, "_")

        return name[:31]

    def _save_model_option_results(
        self,
        results: dict,
        filename: str,
    ) -> None:
        """
        Save successful model-option results.

        For .xlsx/.xls, each successful option is written to a separate sheet.
        For .json, successful options are written as one keyed dictionary.
        For .csv, each successful option is written to a separate CSV file.
        """
        ext = filename.split(".")[-1].lower()
        base = ".".join(filename.split(".")[:-1])

        if ext == "json":
            import json

            with open(filename, "w") as f:
                json.dump(results, f, indent=4)

            return

        if ext in {"xlsx", "xls"}:
            import pandas as pd

            with pd.ExcelWriter(filename) as writer:
                for option_name, records in results.items():
                    df = pd.DataFrame(records)
                    df.to_excel(
                        writer,
                        sheet_name=self._safe_sheet_name(option_name),
                        index=False,
                    )

            return

        if ext == "csv":
            import pandas as pd

            for option_name, records in results.items():
                df = pd.DataFrame(records)
                df.to_csv(
                    f"{base}_{option_name}.csv",
                    index=False,
                )

            return

        raise ValueError(
            "Unsupported file extension. Use .csv, .json, or .xlsx"
        )

    def _format_records_for_return(
        self,
        records: list[dict],
        return_type: str,
    ):
        """
        Convert raw save records to the requested return type.
        """
        return_type = return_type.lower()

        if return_type == "dict":
            return records

        if return_type == "dataframe":
            import pandas as pd
            return pd.DataFrame(records)

        raise ValueError("return_type must be 'dict' or 'dataframe'")

    # ------------------------------------------------------------------
    # Static evaluation path
    # ------------------------------------------------------------------

    def _static_evaluate_once(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        state_max_passes: int = 20,
        state_tolerance: float = 1e-10,
    ):
        """
        Evaluate the currently built network once without model switching.
        """
        start_time = time.perf_counter()

        self.network.pre_evaluation()
        self._refresh_runtime_cache()
        self.evaluate_network_states(
            max_passes=state_max_passes,
            tolerance=state_tolerance,
        )

        elapsed_time = time.perf_counter() - start_time

        solution = self.network.save(
            filename=filename,
            return_type=return_type,
        )

        self._last_success_kind = "static"
        self._last_static_elapsed_time = elapsed_time
        self._last_solve_diagnostics = None

        return solution

    def static_evaluate(
        self,
        model: str | None = None,
        evaluate_all_model_options: bool = False,
        filename: str | None = None,
        return_type: str = "dict",
        verbose: bool = False,
        print_solution: bool = False,
        state_max_passes: int = 20,
        state_tolerance: float = 1e-10,
    ):

        """
        Evaluate the network without performing a nonlinear solve.

        `static_evaluate()` propagates component states and evaluates derived
        quantities using the currently assigned values of all States.

        No iteration variables are modified and no residual equations are solved.

        This mode is useful for:

        * explicit calculations
        * property lookups
        * model validation
        * diagnostic checks
        * post-processing solved networks

        Parameters
        ----------
        model : str, optional
            Model to evaluate. If omitted, all registered Models use their
            currently active option.

        evaluate_all_model_options : bool, optional
            If True, evaluates every option in the selected Model and returns
            successful results.

        filename : str, optional
            Output file used to save results.

        return_type : {"dict", "dataframe"}, optional
            Format of returned solution data.

        verbose : bool, optional
            Print evaluation diagnostics.

        print_solution : bool, optional
            Print exported network values.

        state_max_passes : int, optional
            Maximum number of state-settling passes.

        state_tolerance : float, optional
            Convergence tolerance for derived-state propagation.

        Returns
        -------
        solution : dict or pandas.DataFrame
            Exported network results.

        Notes
        -----
        State propagation is still performed during static evaluation.

        Components may depend on other components through derived States:

            ``composition → density → mass_flow``

        Multiple evaluation passes are therefore executed until all derived
        quantities settle.

        Unlike `solve()`, no iteration variables are modified.

        Examples
        --------
        Evaluate a network once:

            ``solver.static_evaluate()``

        Print results:

            ``solver.static_evaluate(print_solution=True)``

        Evaluate all options of a model:

            ``solver.static_evaluate(``
            ``    model="InjectorModel",``
            ``    evaluate_all_model_options=True``
            ``)``
        """
        selected_model = self._get_model(model)

        # --------------------------------------------------------------
        # Normal static evaluation: models behave like normal components.
        # --------------------------------------------------------------
        if selected_model is None:
            self._build_unbuilt_models()

            solution = self._static_evaluate_once(
                filename=filename,
                return_type=return_type,
                state_max_passes=state_max_passes,
                state_tolerance=state_tolerance,
            )

            self._print_last_success(
                verbose=verbose,
                print_solution=print_solution,
            )

            return solution

        # Other Models still need concrete components, but only the selected
        # Model is swept or retried.
        self._build_unbuilt_models(exclude_model=selected_model)

        # --------------------------------------------------------------
        # Try options until the first successful static evaluation.
        # --------------------------------------------------------------
        if not evaluate_all_model_options:
            failures = []

            for option_name in selected_model.order:
                selected_model.replace(option_name)

                try:
                    solution = self._static_evaluate_once(
                        filename=filename,
                        return_type=return_type,
                        state_max_passes=state_max_passes,
                        state_tolerance=state_tolerance,
                    )

                except Exception as error:
                    self._record_model_failure(
                        failures,
                        error,
                        model=selected_model,
                    )
                    continue

                self._print_model_failures(failures)
                self._print_last_success(
                    verbose=verbose,
                    print_solution=print_solution,
                )

                return solution

            self._print_model_failures(failures)
            raise RuntimeError(
                f"All options failed for model {selected_model.name!r}."
            )

        # --------------------------------------------------------------
        # Attempt every option in the selected Model.
        # --------------------------------------------------------------
        results = {}
        raw_results = {}
        failures = []
        last_success_option = None

        for option_name in selected_model.order:
            selected_model.replace(option_name)

            try:
                self._static_evaluate_once(
                    filename=None,
                    return_type="dict",
                    state_max_passes=state_max_passes,
                    state_tolerance=state_tolerance,
                )

            except Exception as error:
                self._record_model_failure(
                    failures,
                    error,
                    model=selected_model,
                )
                continue

            records = self.network.save(
                filename=None,
                return_type="dict",
            )

            raw_results[option_name] = records
            results[option_name] = self._format_records_for_return(
                records,
                return_type,
            )
            last_success_option = option_name

        if last_success_option is None:
            self._print_model_failures(failures)
            raise RuntimeError(
                f"All options failed for model {selected_model.name!r}."
            )

        # Leave the network and terminal output on the last successful option.
        selected_model.replace(last_success_option)
        self._static_evaluate_once(
            filename=None,
            return_type="dict",
            state_max_passes=state_max_passes,
            state_tolerance=state_tolerance,
        )

        if filename is not None:
            self._save_model_option_results(
                raw_results,
                filename,
            )

        self._print_model_failures(failures)
        self._print_last_success(
            verbose=verbose,
            print_solution=print_solution,
        )

        return results

    # ------------------------------------------------------------------
    # Main nonlinear solve
    # ------------------------------------------------------------------

    def _validate_solve_settings(
        self,
        solver_method: str,
        jacobian_method: str,
        rtol: float,
        state_max_passes: int,
        state_tolerance: float,
    ) -> tuple[str, str]:
        """
        Validate solve settings and return normalized method names.
        """
        method = solver_method.lower()
        valid_solver_methods = ("trf", "dogbox", "lm")

        if method not in valid_solver_methods:
            raise ValueError(
                "solver_method must be one of "
                f"{valid_solver_methods}. "
                f"Got '{solver_method}'."
            )

        jac = jacobian_method.lower()
        valid_jac_methods = ("2-point", "3-point")

        if jac not in valid_jac_methods:
            raise ValueError(
                "jacobian_method must be one of "
                f"{valid_jac_methods}. "
                f"Got '{jacobian_method}'."
            )

        if rtol <= 0.0:
            raise ValueError(
                f"Residual tolerance (rtol) must be positive. "
                f"Got {rtol}"
            )

        if state_max_passes <= 0:
            raise ValueError(
                f"state_max_passes must be positive. "
                f"Got {state_max_passes}"
            )

        if state_tolerance <= 0.0:
            raise ValueError(
                f"state_tolerance must be positive. "
                f"Got {state_tolerance}"
            )

        return method, jac

    def _solve_once(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        solver_method: str = "trf",
        jacobian_method: str = "3-point",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        rtol: float = 1e-2,
        state_max_passes: int = 20,
        state_tolerance: float = 1e-10,
    ):
        """
        Solve the currently built network once without model switching.
        """
        method, jac = self._validate_solve_settings(
            solver_method=solver_method,
            jacobian_method=jacobian_method,
            rtol=rtol,
            state_max_passes=state_max_passes,
            state_tolerance=state_tolerance,
        )

        # Store state-settling controls for residual(), since scipy only calls
        # residual(x) and does not pass solve() keyword arguments through.
        self._state_max_passes = state_max_passes
        self._state_tolerance = state_tolerance

        # One-time component setup.
        self.network.pre_evaluation()
        cache = self._refresh_runtime_cache()

        # Initial solver vector.
        x0 = np.array(
            cache.iteration_values,
            dtype=float,
        )

        # Initial state propagation and residual vector.
        self.evaluate_network_states(
            max_passes=state_max_passes,
            tolerance=state_tolerance,
        )

        r0 = cache.collect_residuals()

        # If there is nothing to solve, use the static path.
        no_iteration_variables = len(x0) == 0
        no_residuals = len(r0) == 0

        if no_iteration_variables and no_residuals:
            return self._static_evaluate_once(
                filename=filename,
                return_type=return_type,
                state_max_passes=state_max_passes,
                state_tolerance=state_tolerance,
            )

        # least_squares requires at least as many residuals as unknowns.
        if len(r0) < len(x0):
            raise ValueError(
                "SteadyState solve requires at least as many "
                "residuals as iteration variables. "
                f"Got {len(x0)} iteration variables "
                f"and {len(r0)} residuals."
            )

        overconstrained = len(r0) > len(x0)

        # Build scipy Bounds object from State metadata.
        bounds = Bounds(
            cache.lower_bounds,
            cache.upper_bounds,
            cache.keep_feasible,
        )

        start_time = time.perf_counter()

        sol = least_squares(
            fun=self.residual,
            x0=x0,
            method=method,
            bounds=bounds,
            x_scale="jac",
            jac=jac,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        elapsed_time = time.perf_counter() - start_time

        final_residual = np.array(
            sol.fun,
            dtype=float,
        )

        max_residual = np.max(np.abs(final_residual))

        if (
            not sol.success
            or max_residual > rtol
        ):
            raise RuntimeError(
                "Steady-state solve failed or converged "
                "to unacceptable residuals.\n"
                f"success = {sol.success}\n"
                f"message = {sol.message}\n"
                f"max |residual| = {max_residual:.3e}\n"
                f"residual tolerance = {rtol:.3e}"
            )

        # Load final solver result into network.
        cache.assign_iteration_values(sol.x)

        # Re-evaluate final derived states using final solver variables.
        self.evaluate_network_states(
            max_passes=state_max_passes,
            tolerance=state_tolerance,
        )

        solution = self.network.save(
            filename=filename,
            return_type=return_type,
        )

        self._last_success_kind = "solve"
        self._last_static_elapsed_time = None
        self._last_solve_diagnostics = {
            "sol": sol,
            "x0": x0,
            "method": method,
            "jac": jac,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "rtol": rtol,
            "overconstrained": overconstrained,
            "elapsed_time": elapsed_time,
        }

        return solution

    def _print_last_success(
        self,
        verbose: bool,
        print_solution: bool,
    ) -> None:
        """
        Print diagnostics for the last successful static evaluation or solve.
        """
        if verbose:
            if getattr(self, "_last_success_kind", None) == "static":
                self._verbose_static_print(
                    elapsed_time=self._last_static_elapsed_time,
                )

            elif getattr(self, "_last_success_kind", None) == "solve":
                diagnostics = self._last_solve_diagnostics
                self._verbose_print(**diagnostics)

        if print_solution:
            self.print_solution()

    def solve(
        self,
        model: str | None = None,
        evaluate_all_model_options: bool = False,
        filename: str | None = None,
        return_type: str = "dict",
        verbose: bool = False,
        static: bool = False,
        print_solution: bool = False,
        solver_method: str = "trf",
        jacobian_method: str = "3-point",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        rtol: float = 1e-2,
        state_max_passes: int = 5,
        state_tolerance: float = 1e-10,
    ):
        """
        Solve the network steady-state nonlinear system.

        `solve()` adjusts network iteration variables until all component and
        balance residual equations satisfy the requested residual tolerance.

        The nonlinear system is solved using
        `scipy.optimize.least_squares()`.

        Parameters
        ----------
        model : str, optional
            Model to solve. If omitted, all registered Models use their
            currently active option.

        evaluate_all_model_options : bool, optional
            If True, attempts every option in the selected Model and returns
            successful solutions.

        filename : str, optional
            Output file used to save results.

        return_type : {"dict", "dataframe"}, optional
            Format of returned solution data.

        verbose : bool, optional
            Print solver diagnostics.

        static : bool, optional
            Perform a static evaluation instead of a nonlinear solve.

        print_solution : bool, optional
            Print exported solution values.

        solver_method : {"trf", "dogbox", "lm"}, optional
            Nonlinear least-squares algorithm.

        jacobian_method : {"2-point", "3-point"}, optional
            Finite-difference Jacobian scheme.

        ftol : float, optional
            Cost-function convergence tolerance.

        xtol : float, optional
            Variable convergence tolerance.

        gtol : float, optional
            Gradient convergence tolerance.

        rtol : float, optional
            Maximum acceptable residual magnitude.

        state_max_passes : int, optional
            Maximum number of state-settling passes.

        state_tolerance : float, optional
            Convergence tolerance for derived-state propagation.

        Returns
        -------
        solution : dict or pandas.DataFrame
            Solved network results.

        Notes
        -----
        The solver seeks:

            ``F(x) = 0``

        where:

            ``x``
                Iteration variables

            ``F``
                Residual equations

        Before each residual evaluation, derived states are repeatedly
        propagated until the network reaches a self-consistent state.

        If a Model is specified, model options may be automatically switched
        until a successful solution is found.

        If all model options fail, a RuntimeError is raised.

        Examples
        --------
        Solve a network:

            ``solver.solve()``

        Print detailed diagnostics:

            ``solver.solve(verbose=True)``

        Save results:

            ``solver.solve(filename="results.xlsx")``

        Solve a specific model:

            ``solver.solve(model="NozzleModel")``

        Evaluate every option of a model:

            ``solver.solve(``
            ``    model="NozzleModel",``
            ``    evaluate_all_model_options=True``
            ``)``
        """
        if static:
            return self.static_evaluate(
                model=model,
                evaluate_all_model_options=evaluate_all_model_options,
                filename=filename,
                return_type=return_type,
                verbose=verbose,
                print_solution=print_solution,
                state_max_passes=state_max_passes,
                state_tolerance=state_tolerance,
            )

        selected_model = self._get_model(model)

        # --------------------------------------------------------------
        # Normal solve: models behave like normal components.
        # --------------------------------------------------------------
        if selected_model is None:
            self._build_unbuilt_models()

            solution = self._solve_once(
                filename=filename,
                return_type=return_type,
                solver_method=solver_method,
                jacobian_method=jacobian_method,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                rtol=rtol,
                state_max_passes=state_max_passes,
                state_tolerance=state_tolerance,
            )

            self._print_last_success(
                verbose=verbose,
                print_solution=print_solution,
            )

            return solution

        # Other Models still need concrete components, but only the selected
        # Model is swept or retried.
        self._build_unbuilt_models(exclude_model=selected_model)

        # --------------------------------------------------------------
        # Try options until the first successful solve.
        # --------------------------------------------------------------
        if not evaluate_all_model_options:
            failures = []

            for option_name in selected_model.order:
                selected_model.replace(option_name)

                try:
                    solution = self._solve_once(
                        filename=filename,
                        return_type=return_type,
                        solver_method=solver_method,
                        jacobian_method=jacobian_method,
                        ftol=ftol,
                        xtol=xtol,
                        gtol=gtol,
                        rtol=rtol,
                        state_max_passes=state_max_passes,
                        state_tolerance=state_tolerance,
                    )

                except Exception as error:
                    self._record_model_failure(
                        failures,
                        error,
                        model=selected_model,
                    )
                    continue

                self._print_model_failures(failures)
                self._print_last_success(
                    verbose=verbose,
                    print_solution=print_solution,
                )

                return solution

            self._print_model_failures(failures)
            raise RuntimeError(
                f"All options failed for model {selected_model.name!r}."
            )

        # --------------------------------------------------------------
        # Attempt every option in the selected Model.
        # --------------------------------------------------------------
        results = {}
        raw_results = {}
        failures = []
        last_success_option = None

        for option_name in selected_model.order:
            selected_model.replace(option_name)

            try:
                self._solve_once(
                    filename=None,
                    return_type="dict",
                    solver_method=solver_method,
                    jacobian_method=jacobian_method,
                    ftol=ftol,
                    xtol=xtol,
                    gtol=gtol,
                    rtol=rtol,
                    state_max_passes=state_max_passes,
                    state_tolerance=state_tolerance,
                )

            except Exception as error:
                self._record_model_failure(
                    failures,
                    error,
                    model=selected_model,
                )
                continue

            records = self.network.save(
                filename=None,
                return_type="dict",
            )

            raw_results[option_name] = records
            results[option_name] = self._format_records_for_return(
                records,
                return_type,
            )
            last_success_option = option_name

        if last_success_option is None:
            self._print_model_failures(failures)
            raise RuntimeError(
                f"All options failed for model {selected_model.name!r}."
            )

        # Leave the network and terminal output on the last successful option.
        selected_model.replace(last_success_option)
        self._solve_once(
            filename=None,
            return_type="dict",
            solver_method=solver_method,
            jacobian_method=jacobian_method,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            rtol=rtol,
            state_max_passes=state_max_passes,
            state_tolerance=state_tolerance,
        )

        if filename is not None:
            self._save_model_option_results(
                raw_results,
                filename,
            )

        self._print_model_failures(failures)
        self._print_last_success(
            verbose=verbose,
            print_solution=print_solution,
        )

        return results
    # ------------------------------------------------------------------
    # Solution printing
    # ------------------------------------------------------------------

    def print_solution(self) -> None:
        """Print exported network state as a rich table."""
        records = self.network.save(return_type="dict")

        table = Table(
            title=f"{self.network.name} Solution",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )

        table.add_column("Component", style="#D84135", no_wrap=True)
        table.add_column("Type", style="#3B629E", no_wrap=True)
        table.add_column("Attribute", style="#fdf0d5", no_wrap=True)
        table.add_column("Value", justify="right")

        for record in records:
            value = record["value"]

            if isinstance(value, float):
                value_text = f"{value:.6g}"
            else:
                value_text = str(value)

            if value_text == "<uninitialized>":
                value_text = "[dim]<uninitialized>[/dim]"

            elif value_text == "<unavailable>":
                value_text = "[red]<unavailable>[/red]"

            table.add_row(
                str(record["component_name"]),
                str(record["component_type"]),
                str(record["attribute"]),
                value_text,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Verbose static evaluation printing
    # ------------------------------------------------------------------

    def _verbose_static_print(
        self,
        elapsed_time: float,
    ) -> None:
        """Print a short summary for static evaluation mode."""
        table = Table(
            title="Static Network Evaluation",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )

        table.add_column("Quantity", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Mode", "Static evaluation")
        table.add_row("Nonlinear solve", "Not performed")
        table.add_row("Evaluation time",f"{elapsed_time:.3f} s")
        table.add_row("Components", str(len(self.network.components)))
        table.add_row(
            "Iteration variables",
            str(len(self.network.collect_all_iteration_variables())),
        )
        table.add_row("Residuals", str(len(self.network.residuals)))

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Shared verbose-label helpers
    # ------------------------------------------------------------------

    def _find_variable_labels(self, target):
        """Find all component/balance attributes that reference a State."""
        labels = []

        for component in self.network.component_list:
            for attr_name, attr_value in component.__dict__.items():
                if attr_value is target:
                    labels.append(
                        f"{component.name}.{attr_name}"
                    )

                if isinstance(attr_value, Composition):
                    for species, state in attr_value.fraction.items():
                        if state is target:
                            labels.append(
                                f"{component.name}.{attr_name}.{species}"
                            )

        for balance in self.network.balance_list:
            for attr_name, attr_value in balance.__dict__.items():
                if attr_value is target:
                    labels.append(
                        f"{balance.name}.{attr_name}"
                    )

        if labels:
            return labels

        return [str(target)]

    def _collect_residual_labels(self) -> list[str]:
        """Collect printed labels for component and balance residuals."""
        residual_labels = []

        for component in self.network.component_list:
            component_residuals = component.residuals

            if isinstance(component_residuals, (list, tuple)):
                for i in range(len(component_residuals)):
                    residual_labels.append(f"{component.name}.residual[{i}]")
            else:
                residual_labels.append(f"{component.name}.residual")

        for balance in self.network.balance_list:
            balance_residuals = balance.residuals

            if isinstance(balance_residuals, (list, tuple)):
                for i in range(len(balance_residuals)):
                    residual_labels.append(f"{balance.name}.residual[{i}]")
            else:
                residual_labels.append(f"{balance.name}.residual")

        return residual_labels

    # ------------------------------------------------------------------
    # Verbose nonlinear solve printing
    # ------------------------------------------------------------------

    def _verbose_print(
        self,
        sol,
        x0: np.ndarray,
        method: str,
        jac: str,
        ftol: float,
        xtol: float,
        gtol: float,
        rtol: float,
        overconstrained: bool = False,
        elapsed_time: float = 0.0,
    ) -> None:
        """Print solver summary, final variables, and final residuals."""

        max_residual = np.max(np.abs(sol.fun))
        rms_residual = np.sqrt(np.mean(sol.fun**2))

        dx = np.array(sol.x, dtype=float) - np.array(x0, dtype=float)

        # Variable adjustment from initial guess to final solution.
        normalized_variable_adjustment = (
            np.abs(dx) / np.maximum(np.abs(sol.x), 1.0)
        )

        max_normalized_variable_adjustment = np.max(
            normalized_variable_adjustment
        )

        # --------------------------------------------------------------
        # Solver summary table.
        # --------------------------------------------------------------
        summary = Table(
            title="Steady-State Solver Summary",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )

        summary.add_column("Quantity", style="bold")
        summary.add_column("Value", justify="right")

        summary.add_row("Success", str(sol.success))
        summary.add_row("Status", str(sol.status))
        summary.add_row("Message", str(sol.message))

        if overconstrained:
            summary.add_row(
                "Warning",
                "System is overconstrained",
                style="yellow",
            )

        summary.add_row("Solver method", method)
        summary.add_row("Jacobian method", jac)
        summary.add_row("Solve time",f"{elapsed_time:.3f} s")
        summary.add_row("Function evaluations", str(sol.nfev))

        if hasattr(sol, "njev") and sol.njev is not None:
            summary.add_row("Jacobian evaluations", str(sol.njev))

        if hasattr(sol, "cost"):
            summary.add_row("Cost", f"{sol.cost:.6e}")

        if hasattr(sol, "optimality"):
            summary.add_row("Optimality", f"{sol.optimality:.3e}")

        summary.add_row("Max |residual|", f"{max_residual:.3e}")
        summary.add_row("RMS residual", f"{rms_residual:.3e}")
        summary.add_row(
            "Max normalized variable adjustment",
            f"{max_normalized_variable_adjustment:.3e}",
        )
        summary.add_row("Residual tolerance", f"{rtol:.3e}")
        summary.add_row("ftol", f"{ftol:.3e}")
        summary.add_row("xtol", f"{xtol:.3e}")
        summary.add_row("gtol", f"{gtol:.3e}")

        # --------------------------------------------------------------
        # Iteration variable table.
        # --------------------------------------------------------------
        variables = Table(
            title="Solution Variables",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )

        variables.add_column("Index", justify="right", style="dim")
        variables.add_column("Variable", style="#fdf0d5")
        variables.add_column("Value", justify="right", style="#D84135")
        variables.add_column("Variable Adjustment", justify="right", style="#3B629E")
        variables.add_column("Normalized Variable Adjustment", justify="right", style="#3B629E")

        variable_labels = [
            self._find_variable_labels(var)
            for var in self.network.collect_all_iteration_variables()
        ]

        for i, val in enumerate(sol.x):
            label = (
                "\n".join(variable_labels[i])
                if i < len(variable_labels)
                else "<unlabeled>"
            )

            variables.add_row(
                f"x[{i}]",
                label,
                f"{val:.6e}",
                f"{dx[i]:+.6e}",
                f"{normalized_variable_adjustment[i]:.3e}",
            )

        # --------------------------------------------------------------
        # Residual table.
        # --------------------------------------------------------------
        residuals = Table(
            title="Residuals",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )

        residuals.add_column("Index", justify="right", style="dim")
        residuals.add_column("Residual", style="#fdf0d5")
        residuals.add_column("Value", justify="right", style="#3B629E")

        residual_labels = self._collect_residual_labels()

        for i, r in enumerate(sol.fun):
            label = (
                residual_labels[i]
                if i < len(residual_labels)
                else "<unlabeled>"
            )

            residuals.add_row(
                f"r[{i}]",
                label,
                f"{r:.6e}",
            )

        self.console.print()
        self.console.print(summary)
        self.console.print(variables)
        self.console.print(residuals)
        self.console.print()

    def _verbose_failure_print(
        self,
        x0: np.ndarray,
        method: str,
        jac: str,
        ftol: float,
        xtol: float,
        gtol: float,
        rtol: float,
        overconstrained: bool = False,
        elapsed_time: float = 0.0,
    ) -> None:
        """
        Print the last attempted solver state when scipy fails before
        returning a solution object.
        """
        x = self._last_debug_x
        r = self._last_debug_residual
        error = self._last_debug_error

        summary = Table(
            title="Steady-State Solver Failure",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )

        summary.add_column("Quantity", style="bold")
        summary.add_column("Value", justify="right")

        summary.add_row("Success", "False")
        summary.add_row("Solver method", method)
        summary.add_row("Jacobian method", jac)
        summary.add_row("Solve time", f"{elapsed_time:.3f} s")

        if overconstrained:
            summary.add_row(
                "Warning",
                "System is overconstrained",
                style="yellow",
            )

        if error is not None:
            summary.add_row("Error type", type(error).__name__)
            summary.add_row("Error", str(error))

        summary.add_row("Residual tolerance", f"{rtol:.3e}")
        summary.add_row("ftol", f"{ftol:.3e}")
        summary.add_row("xtol", f"{xtol:.3e}")
        summary.add_row("gtol", f"{gtol:.3e}")

        self.console.print()
        self.console.print(summary)

        if x is not None:
            dx = np.array(x, dtype=float) - np.array(x0, dtype=float)

            # Variable adjustment from initial guess to last attempted iterate.
            normalized_variable_adjustment = (
                np.abs(dx) / np.maximum(np.abs(x), 1.0)
            )

            variables = Table(
                title="Last Attempted Solution Variables",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold",
            )

            variables.add_column("Index", justify="right", style="dim")
            variables.add_column("Variable", style="#fdf0d5")
            variables.add_column("Value", justify="right", style="#D84135")
            variables.add_column("Variable Adjustment", justify="right", style="#3B629E")
            variables.add_column("Normalized Variable Adjustment", justify="right", style="#3B629E")

            variable_labels = [
                self._find_variable_labels(var)
                for var in self.network.collect_all_iteration_variables()
            ]

            for i, val in enumerate(x):
                label = (
                    "\n".join(variable_labels[i])
                    if i < len(variable_labels)
                    else "<unlabeled>"
                )

                variables.add_row(
                    f"x[{i}]",
                    label,
                    f"{val:.6e}",
                    f"{dx[i]:+.6e}",
                    f"{normalized_variable_adjustment[i]:.3e}",
                )

            self.console.print(variables)

        if r is not None:
            residuals = Table(
                title="Last Available Residuals",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold",
            )

            residuals.add_column("Index", justify="right", style="dim")
            residuals.add_column("Residual", style="#fdf0d5")
            residuals.add_column("Value", justify="right", style="#3B629E")

            residual_labels = self._collect_residual_labels()

            for i, value in enumerate(r):
                label = (
                    residual_labels[i]
                    if i < len(residual_labels)
                    else "<unlabeled>"
                )

                residuals.add_row(
                    f"r[{i}]",
                    label,
                    f"{value:.6e}",
                )

            self.console.print(residuals)

        self.console.print()