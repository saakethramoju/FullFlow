"""Model-option orchestration for transient solving."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from fullflow.System.State import is_state_like

from fullflow.Exports.HDF5 import run_group_path, safe_group_name, write_failures
from fullflow.System.State import State
from fullflow.Solvers.steady_state.models import ModelFailure, ModelManager
from .results import format_records


class NetworkStateSnapshot:
    """Capture and restore State values before independent transient runs.

    State values can themselves contain references to other State objects. This
    happens in components such as Map, where the ``inputs`` State stores a
    dictionary of live input States.  The snapshot must preserve those State
    references instead of deep-copying them into disconnected clones; otherwise
    model-option transient runs can freeze maps at their initial input values.
    """

    @staticmethod
    def _snapshot_value(value: Any) -> Any:
        if is_state_like(value):
            return value

        if isinstance(value, dict):
            return {
                NetworkStateSnapshot._snapshot_value(key): NetworkStateSnapshot._snapshot_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [NetworkStateSnapshot._snapshot_value(item) for item in value]

        if isinstance(value, tuple):
            return tuple(NetworkStateSnapshot._snapshot_value(item) for item in value)

        if isinstance(value, set):
            return {NetworkStateSnapshot._snapshot_value(item) for item in value}

        if isinstance(value, frozenset):
            return frozenset(
                NetworkStateSnapshot._snapshot_value(item)
                for item in value
            )

        try:
            return deepcopy(value)
        except Exception:
            return value

    def __init__(self, states: list[State]) -> None:
        self.states = states
        self.values = {
            state: (
                self._snapshot_value(state._value),
                self._snapshot_value(state._previous),
                self._snapshot_value(state._second_previous),
            )
            for state in states
        }

    @classmethod
    def capture(cls, network) -> "NetworkStateSnapshot":
        states: list[State] = []
        seen_objects: set[int] = set()
        seen_states: set[int] = set()

        def visit(value: Any) -> None:
            value_id = id(value)
            if value_id in seen_objects:
                return
            seen_objects.add(value_id)

            if isinstance(value, State):
                if value_id not in seen_states:
                    seen_states.add(value_id)
                    states.append(value)
                return

            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return

            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    visit(item)
                return

            if hasattr(value, "__dict__"):
                for item in vars(value).values():
                    visit(item)

        visit(network)
        return cls(states)

    def restore(self) -> None:
        for state, (value, previous, second_previous) in self.values.items():
            state._value = self._snapshot_value(value)
            state._previous = self._snapshot_value(previous)
            state._second_previous = self._snapshot_value(second_previous)


class TransientModelOptionRunner:
    """Run transient solves through optional model selection.

    Each model option is a complete independent transient run.  All evaluated
    options start from the same captured initial State values.
    """

    def __init__(self, network, model_manager: ModelManager, printer) -> None:
        self.network = network
        self.model_manager = model_manager
        self.printer = printer

    @staticmethod
    def _metadata(*, model_name: str | None = None, option_name: str | None = None, evaluate_all: bool = False) -> dict[str, Any]:
        return {
            "solve_type": "transient",
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
        run_once: Callable[..., Any],
    ):
        selected_model = self.model_manager.get(model)

        if selected_model is None:
            self.model_manager.build_unbuilt()
            return run_once(
                filename=filename,
                return_type=return_type,
                group_path=run_group_path("transient"),
                metadata=self._metadata(),
                verbose_override=verbose,
            )

        self.model_manager.build_unbuilt(exclude_model=selected_model)
        snapshot = NetworkStateSnapshot.capture(self.network)

        if evaluate_all_model_options:
            return self._run_all_options(
                selected_model,
                snapshot,
                filename=filename,
                return_type=return_type,
                verbose=verbose,
                run_once=run_once,
            )

        return self._run_first_working_option(
            selected_model,
            snapshot,
            filename=filename,
            return_type=return_type,
            verbose=verbose,
            run_once=run_once,
        )

    def _run_first_working_option(
        self,
        selected_model,
        snapshot: NetworkStateSnapshot,
        *,
        filename: str | None,
        return_type: str,
        verbose: bool,
        run_once: Callable[..., Any],
    ):
        failures: list[ModelFailure] = []

        for option_name in selected_model.order:
            selected_model.replace(option_name)
            snapshot.restore()
            try:
                records = run_once(
                    filename=filename,
                    return_type=return_type,
                    group_path=run_group_path("transient", model_name=selected_model.name, option_name=option_name),
                    metadata=self._metadata(model_name=selected_model.name, option_name=option_name),
                    verbose_override=verbose,
                )
            except Exception as error:
                failures.append(ModelFailure.from_error(self.network, error, selected_model))
                snapshot.restore()
                continue

            if filename is not None and failures:
                write_failures(
                    filename,
                    failures,
                    group_path=(
                        f"{safe_group_name(self.network.name)}/transient/runs/"
                        f"{safe_group_name(selected_model.name)}/failures"
                    ),
                )
            self.printer.print_model_failures(failures)
            return records

        self.printer.print_model_failures(failures)
        raise RuntimeError(f"All options failed for model {selected_model.name!r}.")

    def _run_all_options(
        self,
        selected_model,
        snapshot: NetworkStateSnapshot,
        *,
        filename: str | None,
        return_type: str,
        verbose: bool,
        run_once: Callable[..., Any],
    ):
        results: dict[str, Any] = {}
        failures: list[ModelFailure] = []
        last_success_option = None

        for option_name in selected_model.order:
            selected_model.replace(option_name)
            snapshot.restore()
            try:
                records = run_once(
                    filename=filename,
                    return_type="dict",
                    group_path=run_group_path("transient", model_name=selected_model.name, option_name=option_name),
                    metadata=self._metadata(
                        model_name=selected_model.name,
                        option_name=option_name,
                        evaluate_all=True,
                    ),
                    verbose_override=False,
                )
            except Exception as error:
                failures.append(ModelFailure.from_error(self.network, error, selected_model))
                snapshot.restore()
                continue

            results[option_name] = format_records(records, return_type)
            last_success_option = option_name

        if last_success_option is None:
            self.printer.print_model_failures(failures)
            raise RuntimeError(f"All options failed for model {selected_model.name!r}.")

        selected_model.replace(last_success_option)
        snapshot.restore()
        run_once(filename=None, return_type="dict", verbose_override=verbose)

        if filename is not None and failures:
            write_failures(
                filename,
                failures,
                group_path=(
                    f"{safe_group_name(self.network.name)}/transient/runs/"
                    f"{safe_group_name(selected_model.name)}/failures"
                ),
            )

        self.printer.print_model_failures(failures)
        return results
