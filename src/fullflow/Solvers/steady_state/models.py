"""Model-option orchestration for steady-state solving.

FullFlow ``Model`` objects let one network try alternative component
implementations, such as choked vs. unchoked flow. This file keeps that fallback
logic out of the public ``SteadyState`` wrapper so the API file stays small and
the model behavior can be debugged in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .results import format_records, save_model_option_results
from .statistics import model_option_statistics_path, statistics_path
from fullflow.Exports.HDF5 import HDF5Target, run_group_path, safe_group_name, write_failures, write_solution
from fullflow.Exceptions import SolverConvergenceError, SolverSetupError


@dataclass(slots=True)
class ModelFailure:
    """Compact record for a model option that failed during evaluation."""

    model: str
    option: str
    error_type: str
    error: str

    @classmethod
    def from_error(cls, network, error: Exception, model: Any = None) -> "ModelFailure":
        """Create a diagnostic record from an exception.

        When ``model`` is omitted, the failure is treated as a whole-network
        setup/evaluation failure rather than one selected model option failure.
        """
        if model is None:
            option = ", ".join(
                f"{candidate.name}={candidate.active_option_name or '<unbuilt>'}"
                for candidate in network.model_list
            ) or "<none>"
            return cls(
                model="<network>",
                option=option,
                error_type=type(error).__name__,
                error=str(error).splitlines()[0],
            )

        return cls(
            model=model.name,
            option=model.active_option_name or "<unbuilt>",
            error_type=type(error).__name__,
            error=str(error).splitlines()[0],
        )


class ModelManager:
    """Small helper for finding and building network ``Model`` objects."""

    def __init__(self, network) -> None:
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.network = network

    def get(self, model: str | Any | None):
        """Resolve a model name/object passed to ``solve(model=...)``.

        ``None`` means the caller is not asking to iterate over a specific
        model. In that case all unbuilt models are simply built with their first
        options before the network is evaluated.
        """
        if model is None:
            return None

        if hasattr(model, "build") and hasattr(model, "available_options"):
            return model

        for candidate in self.network.model_list:
            if candidate.name == model:
                return candidate

        raise SolverSetupError(
            f"Unknown model {model!r}. Available models are "
            f"{[m.name for m in self.network.model_list]}."
        )

    def build_unbuilt(self, exclude_model=None) -> None:
        """Build every unbuilt model, optionally leaving one model unbuilt."""
        for model in self.network.model_list:
            if model is exclude_model:
                continue
            if model.active_component is None:
                model.build()

    def active_summary(self) -> str:
        """Human-readable list of active model options."""
        if not self.network.model_list:
            return "<none>"
        return ", ".join(
            f"{model.name}={model.active_option_name or '<unbuilt>'}"
            for model in self.network.model_list
        )


class ModelOptionRunner:
    """Run static/solve operations through optional model selection.

    Model-option runs are exported as complete independent run groups under::

        /<network>/steady_state/runs/base
        /<network>/steady_state/runs/<model>/<option>
    """

    def __init__(
        self,
        network,
        model_manager: ModelManager,
        printer,
        success_printer: Callable[[bool], None],
    ) -> None:
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.network = network
        self.model_manager = model_manager
        self.printer = printer
        self.success_printer = success_printer

    @staticmethod
    def _metadata(*, model_name: str | None = None, option_name: str | None = None, evaluate_all: bool = False) -> dict[str, Any]:
        return {
            "solve_type": "steady_state",
            "run_type": "base" if model_name is None else "model_option",
            "model_name": "" if model_name is None else model_name,
            "option_name": "" if option_name is None else option_name,
            "evaluate_all_model_options": bool(evaluate_all),
        }

    def run(
        self,
        *,
        model: str | Any | None,
        evaluate_all_model_options: bool,
        filename: str | None,
        return_type: str,
        verbose: bool,
        statistics: bool,
        run_once: Callable[..., Any],
    ):
        """Run one operation with optional model-option fallback/evaluation."""
        selected_model = self.model_manager.get(model)

        if selected_model is None:
            return self._run_without_selected_model(
                filename=filename,
                return_type=return_type,
                verbose=verbose,
                statistics=statistics,
                run_once=run_once,
            )

        self.model_manager.build_unbuilt(exclude_model=selected_model)

        if evaluate_all_model_options:
            return self._run_all_options(
                selected_model,
                filename=filename,
                return_type=return_type,
                verbose=verbose,
                statistics=statistics,
                run_once=run_once,
            )

        return self._run_first_working_option(
            selected_model,
            filename=filename,
            return_type=return_type,
            verbose=verbose,
            statistics=statistics,
            run_once=run_once,
        )

    def _run_without_selected_model(
        self,
        *,
        filename: str | None,
        return_type: str,
        verbose: bool,
        statistics: bool,
        run_once: Callable[..., Any],
    ):
        """Build default model options and run the operation once."""
        self.model_manager.build_unbuilt()
        group_path = run_group_path("steady_state")
        solution = run_once(
            filename=filename,
            return_type=return_type,
            statistics_filename=(statistics_path(filename, self.network.name) if statistics and filename is not None else None),
            group_path=group_path,
            metadata=self._metadata(),
        )
        self.success_printer(verbose)
        return solution

    def _run_first_working_option(
        self,
        selected_model,
        *,
        filename: str | None,
        return_type: str,
        verbose: bool,
        statistics: bool,
        run_once: Callable[..., Any],
    ):
        """Try model options in order and return the first successful result."""
        failures: list[ModelFailure] = []

        for option_name in selected_model.order:
            selected_model.replace(option_name)
            group_path = run_group_path(
                "steady_state",
                model_name=selected_model.name,
                option_name=option_name,
            )
            try:
                statistics_target = (
                    model_option_statistics_path(filename, selected_model.name, option_name, self.network.name)
                    if statistics and filename is not None
                    else None
                )
                solution = run_once(
                    filename=filename,
                    return_type=return_type,
                    statistics_filename=statistics_target,
                    group_path=group_path,
                    metadata=self._metadata(
                        model_name=selected_model.name,
                        option_name=option_name,
                        evaluate_all=False,
                    ),
                )
            except Exception as error:
                failures.append(ModelFailure.from_error(self.network, error, selected_model))
                continue

            if filename is not None and failures:
                write_failures(
                    filename,
                    failures,
                    group_path=(
                        f"{safe_group_name(self.network.name)}/steady_state/runs/"
                        f"{safe_group_name(selected_model.name)}/failures"
                    ),
                )

            self.printer.print_model_failures(failures)
            self.success_printer(verbose)
            return solution

        self.printer.print_model_failures(failures)
        raise SolverConvergenceError(f"All options failed for model {selected_model.name!r}.")

    def _run_all_options(
        self,
        selected_model,
        *,
        filename: str | None,
        return_type: str,
        verbose: bool,
        statistics: bool,
        run_once: Callable[..., Any],
    ):
        """Evaluate every model option and return a dict keyed by option name."""
        results: dict[str, Any] = {}
        failures: list[ModelFailure] = []
        last_success_option = None

        for option_name in selected_model.order:
            selected_model.replace(option_name)
            group_path = run_group_path(
                "steady_state",
                model_name=selected_model.name,
                option_name=option_name,
            )
            try:
                records = run_once(
                    filename=filename,
                    return_type="dict",
                    statistics_filename=(
                        model_option_statistics_path(filename, selected_model.name, option_name, self.network.name)
                        if statistics and filename is not None
                        else None
                    ),
                    group_path=group_path,
                    metadata=self._metadata(
                        model_name=selected_model.name,
                        option_name=option_name,
                        evaluate_all=True,
                    ),
                )
            except Exception as error:
                failures.append(ModelFailure.from_error(self.network, error, selected_model))
                continue

            results[option_name] = format_records(records, return_type)
            last_success_option = option_name

        if last_success_option is None:
            self.printer.print_model_failures(failures)
            raise SolverConvergenceError(f"All options failed for model {selected_model.name!r}.")

        # Re-run the last successful option without exporting so the live network
        # values match the option left active after this method returns.
        selected_model.replace(last_success_option)
        run_once(filename=None, return_type="dict", statistics_filename=None)

        if filename is not None and failures:
            write_failures(
                filename,
                failures,
                group_path=(
                    f"{safe_group_name(self.network.name)}/steady_state/runs/"
                    f"{safe_group_name(selected_model.name)}/failures"
                ),
            )

        self.printer.print_model_failures(failures)
        self.success_printer(verbose)
        return results
