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
from .statistics import model_option_statistics_path
from fullflow.Exports.HDF5 import HDF5Target, safe_group_name, write_failures, write_solution


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

        raise ValueError(
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

    The solver has two layers of execution:

    1. A ``run_once`` callable that assumes the network already contains the
       desired concrete components.
    2. This wrapper, which builds model options, retries failures, optionally
       evaluates every option, and then delegates to ``run_once``.

    Keeping this policy here lets ``SteadyState.solve()`` stay focused on API
    arguments while keeping model fallback behavior shared by ``solve`` and
    ``static_evaluate``.
    """

    def __init__(
        self,
        network,
        model_manager: ModelManager,
        printer,
        success_printer: Callable[[bool], None],
    ) -> None:
        self.network = network
        self.model_manager = model_manager
        self.printer = printer
        self.success_printer = success_printer

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
        solution = run_once(
            filename=filename,
            return_type=return_type,
            statistics_filename=filename if statistics else None,
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
            try:
                statistics_target = (
                    model_option_statistics_path(filename, selected_model.name, option_name)
                    if statistics and filename is not None
                    else None
                )
                solution = run_once(
                    filename=None,
                    return_type=return_type,
                    statistics_filename=statistics_target,
                )
            except Exception as error:
                failures.append(ModelFailure.from_error(self.network, error, selected_model))
                continue

            if filename is not None:
                records = self.network.save(filename=None, return_type="dict")
                option_group = (
                    f"solutions/model_options/{safe_group_name(selected_model.name)}/"
                    f"{safe_group_name(option_name)}"
                )
                write_solution(
                    HDF5Target(filename, option_group),
                    records,
                    network_name=self.network.name,
                    models=self.network.model_list,
                )
                write_solution(
                    filename,
                    records,
                    network_name=self.network.name,
                    models=self.network.model_list,
                )
                if failures:
                    write_failures(
                        filename,
                        failures,
                        group_path=f"diagnostics/model_options/{safe_group_name(selected_model.name)}/failures",
                    )

            self.printer.print_model_failures(failures)
            self.success_printer(verbose)
            return solution

        self.printer.print_model_failures(failures)
        raise RuntimeError(f"All options failed for model {selected_model.name!r}.")

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
        """Evaluate every model option and return a dict keyed by option name.

        The network is left on the last successful option so subsequent
        ``network.save()`` calls show a valid concrete model rather than a
        failed/skipped option.
        """
        results: dict[str, Any] = {}
        raw_results: dict[str, list[dict[str, Any]]] = {}
        failures: list[ModelFailure] = []
        last_success_option = None

        for option_name in selected_model.order:
            selected_model.replace(option_name)
            try:
                run_once(
                    filename=None,
                    return_type="dict",
                    statistics_filename=(
                        model_option_statistics_path(filename, selected_model.name, option_name)
                        if statistics and filename is not None
                        else None
                    ),
                )
            except Exception as error:
                failures.append(ModelFailure.from_error(self.network, error, selected_model))
                continue

            records = self.network.save(filename=None, return_type="dict")
            raw_results[option_name] = records
            results[option_name] = format_records(records, return_type)
            last_success_option = option_name

        if last_success_option is None:
            self.printer.print_model_failures(failures)
            raise RuntimeError(f"All options failed for model {selected_model.name!r}.")

        # Re-run the last successful option so the live network values match the
        # option left active after this method returns.
        selected_model.replace(last_success_option)
        run_once(filename=None, return_type="dict", statistics_filename=None)

        if filename is not None:
            save_model_option_results(
                raw_results,
                filename,
                model_name=selected_model.name,
                network_name=self.network.name,
            )
            records = self.network.save(filename=None, return_type="dict")
            write_solution(
                filename,
                records,
                network_name=self.network.name,
                models=self.network.model_list,
            )
            if failures:
                write_failures(
                    filename,
                    failures,
                    group_path=f"diagnostics/model_options/{safe_group_name(selected_model.name)}/failures",
                )

        self.printer.print_model_failures(failures)
        self.success_printer(verbose)
        return results
