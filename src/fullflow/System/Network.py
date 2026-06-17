from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .Composition import Composition
from .State import is_assignable_state_like, is_state_like, resolve_value

if TYPE_CHECKING:
    from fullflow.System import Balance, Component, State


class Network:
    """Container for FullFlow components, balances, models, and shared states."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.component_list: list[Component] = []
        self.balance_list: list[Balance] = []
        self.tracked_state_list: list[tuple[str, State]] = []
        self.model_list: list[Any] = []

        # Incremented whenever the network structure changes. Solver runtime
        # plans use this to rebuild only when needed.
        self._version = 0

        self._iteration_cache_valid = False
        self._component_iteration_variables: tuple[State, ...] = ()
        self._balance_iteration_variables: tuple[State, ...] = ()
        self._all_iteration_variables: tuple[State, ...] = ()
        self._iteration_variable_labels: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        """Monotonic network-structure version for solver/runtime caches."""
        return self._version

    def mark_structure_changed(self) -> None:
        """Invalidate cached network metadata after structural edits."""
        self._version += 1
        self._iteration_cache_valid = False

    def add_component(self, component: Component) -> None:
        if component not in self.component_list:
            self.component_list.append(component)
            self.mark_structure_changed()

    def remove_component(self, component: Component) -> None:
        if component not in self.component_list:
            raise ValueError(
                f"Component {component.name!r} is not registered with network {self.name!r}."
            )
        self.component_list.remove(component)
        self.mark_structure_changed()

    def add_balance(self, balance: Balance) -> None:
        if balance not in self.balance_list:
            self.balance_list.append(balance)
            self.mark_structure_changed()

    def add_model(self, model) -> None:
        if model not in self.model_list:
            self.model_list.append(model)
            self.mark_structure_changed()

    def track(self, name: str, state: State) -> State:
        self.tracked_state_list.append((name, state))
        self.mark_structure_changed()
        return state

    @property
    def components(self) -> list[str]:
        return [component.name for component in self.component_list]

    @property
    def balances(self) -> list[str]:
        return [balance.name for balance in self.balance_list]

    @property
    def models(self) -> list[str]:
        return [model.name for model in self.model_list]

    # ------------------------------------------------------------------
    # Iteration variable metadata
    # ------------------------------------------------------------------

    def refresh_iteration_cache(self) -> None:
        self._iteration_cache_valid = False
        self._ensure_iteration_cache()

    def _state_label(self, owner, state: State) -> str:
        for attr_name, attr_value in owner.__dict__.items():
            if attr_value is state:
                return f"{owner.name}:{attr_name}"
            if isinstance(attr_value, Composition):
                for species, species_state in attr_value.fraction.items():
                    if species_state is state:
                        return f"{owner.name}:{attr_name}.{species}"
        return f"{owner.name}:<unknown>"

    def _ensure_iteration_cache(self) -> None:
        if self._iteration_cache_valid:
            return

        component_vars: list[State] = []
        balance_vars: list[State] = []
        labels: list[str] = []

        for component in self.component_list:
            for state in component.iteration_variables:
                if not is_assignable_state_like(state):
                    raise TypeError(
                        f"{component.name}: iteration variable must be an "
                        "assignable State-like object."
                    )
                component_vars.append(state)
                labels.append(self._state_label(component, state))

        for balance in self.balance_list:
            for state in balance.iteration_variables:
                if not is_assignable_state_like(state):
                    raise TypeError(
                        f"{balance.name}: iteration variable must be an "
                        "assignable State-like object."
                    )
                balance_vars.append(state)
                labels.append(self._state_label(balance, state))

        self._validate_no_iteration_overlap(component_vars, balance_vars)

        self._component_iteration_variables = tuple(component_vars)
        self._balance_iteration_variables = tuple(balance_vars)
        self._all_iteration_variables = tuple(component_vars + balance_vars)
        self._iteration_variable_labels = tuple(labels)
        self._iteration_cache_valid = True

    @property
    def iteration_variable_labels(self) -> list[str]:
        self._ensure_iteration_cache()
        return list(self._iteration_variable_labels)

    @property
    def component_iteration_variables(self) -> list[State]:
        self._ensure_iteration_cache()
        return list(self._component_iteration_variables)

    @property
    def balance_iteration_variables(self) -> list[State]:
        self._ensure_iteration_cache()
        return list(self._balance_iteration_variables)

    @property
    def iteration_variable_states(self) -> list[State]:
        self._ensure_iteration_cache()
        return list(self._all_iteration_variables)

    @property
    def lower_bounds(self) -> list[float]:
        return [state.lower_bound for state in self.iteration_variable_states]

    @property
    def upper_bounds(self) -> list[float]:
        return [state.upper_bound for state in self.iteration_variable_states]

    @property
    def has_bounds(self) -> bool:
        return any(state.has_bounds for state in self.iteration_variable_states)

    @property
    def keep_feasible(self) -> list[bool]:
        return [state.keep_feasible for state in self.iteration_variable_states]

    @property
    def iteration_values(self) -> list[float]:
        return [state.numeric_value for state in self.iteration_variable_states]

    def assign_iteration_values(self, iteration_values: list[float]) -> None:
        variables = self.iteration_variable_states
        if len(iteration_values) != len(variables):
            raise ValueError(
                f"Length mismatch: got {len(iteration_values)} iteration values "
                f"but expected {len(variables)}."
            )

        for value, state in zip(iteration_values, variables):
            state.value = value

    def _validate_no_iteration_overlap(
        self,
        component_vars: list[State] | None = None,
        balance_vars: list[State] | None = None,
    ) -> None:
        component_vars = component_vars if component_vars is not None else self.component_iteration_variables
        balance_vars = balance_vars if balance_vars is not None else self.balance_iteration_variables

        component_ids = {id(state) for state in component_vars}
        balance_ids = {id(state) for state in balance_vars}
        overlap_ids = component_ids & balance_ids

        if overlap_ids:
            raise ValueError(self._format_iteration_overlap_error(overlap_ids))

    def _format_iteration_overlap_error(self, overlap_ids: set[int]) -> str:
        component_names = []
        balance_names = []

        for component in self.component_list:
            for state in component.iteration_variables:
                if id(state) in overlap_ids:
                    component_names.append(self._state_label(component, state))

        for balance in self.balance_list:
            for state in balance.iteration_variables:
                if id(state) in overlap_ids:
                    balance_names.append(self._state_label(balance, state))

        lines = [
            "Iteration variable overlap detected.",
            "",
            "Balance iteration variables cannot be the same as component iteration variables.",
            "A State used as a Balance solve variable must not also appear in a Component iteration_variables list.",
            "",
            "Overlapping component variables:",
            *[f"  - {name}" for name in sorted(set(component_names))],
            "",
            "Conflicting balance variables:",
            *[f"  - {name}" for name in sorted(set(balance_names))],
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Residuals and evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_residuals(residual_source) -> list[float]:
        if residual_source is None:
            return []
        if isinstance(residual_source, (list, tuple)):
            values = residual_source
        else:
            values = (residual_source,)

        residuals = []
        for value in values:
            residuals.append(float(resolve_value(value)))
        return residuals

    @property
    def residuals(self) -> list[float]:
        residuals: list[float] = []

        for component in self.component_list:
            try:
                residuals.extend(self._flatten_residuals(component.residuals))
            except Exception as error:
                original = str(error).splitlines()[0]
                raise RuntimeError(
                    f"Failed while evaluating residuals for component `{component.name}` "
                    f"of type `{type(component).__name__}`.\n\n"
                    "A State used inside this component's residual equations is probably unassigned.\n\n"
                    "Likely fixes:\n"
                    "  - Give the missing State an initial value\n"
                    "  - Connect it to another component that computes it\n"
                    "  - Make it an iteration variable\n"
                    "  - If this is a static/transient-only quantity, do not use it in steady-state residuals\n\n"
                    f"Original error: {original}"
                ) from None

        for balance in self.balance_list:
            residuals.extend(self._flatten_residuals(balance.residuals))

        return residuals

    def pre_evaluation(self) -> None:
        for component in self.component_list:
            component.pre_evaluation()

    def evaluate_states(self) -> None:
        for component in self.component_list:
            component.evaluate_states()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    @classmethod
    def _safe_value(cls, value: Any) -> Any:
        if is_state_like(value):
            if not value.is_assigned:
                return "<uninitialized>"
            try:
                return value.value
            except Exception:
                return "<unavailable>"
        return value

    def _append_record(
        self,
        records: list[dict[str, Any]],
        owner_name: str,
        owner_type: str,
        attribute: str,
        value: Any,
    ) -> None:
        records.append({
            "component_name": owner_name,
            "component_type": owner_type,
            "attribute": attribute,
            "value": value,
        })

    def _export_owner(
        self,
        owner,
        records: list[dict[str, Any]],
        ignored_attributes: set[str],
    ) -> None:
        owner_name = owner.name
        owner_type = owner.__class__.__name__

        for attr_name, attr_value in owner.__dict__.items():
            if attr_name in ignored_attributes or attr_name.startswith("_"):
                continue

            if isinstance(attr_value, Composition):
                if attr_value.is_assigned:
                    for species, state in attr_value.fraction.items():
                        self._append_record(
                            records,
                            owner_name,
                            owner_type,
                            f"{attr_name}.{species}",
                            self._safe_value(state),
                        )
                else:
                    self._append_record(
                        records,
                        owner_name,
                        owner_type,
                        attr_name,
                        "<uninitialized>",
                    )
                continue

            self._append_record(
                records,
                owner_name,
                owner_type,
                attr_name,
                self._safe_value(attr_value),
            )

    def save(self, filename: str | None = None, return_type: str = "dict"):
        """Export component, balance, and tracked state values."""
        return_type = return_type.lower()
        records: list[dict[str, Any]] = []

        for component in self.component_list:
            ignored = {"name", "network"} | component.ignored_export_attributes
            self._export_owner(component, records, ignored)

        for balance in self.balance_list:
            self._export_owner(balance, records, {"name", "network"})

        for name, state in self.tracked_state_list:
            self._append_record(
                records,
                self.name,
                "TrackedState",
                name,
                self._safe_value(state),
            )

        if return_type == "dict":
            result = records
        elif return_type == "dataframe":
            import pandas as pd
            result = pd.DataFrame(records)
        else:
            raise ValueError("return_type must be 'dict' or 'dataframe'.")

        if filename is not None:
            path = Path(filename)
            ext = path.suffix.lower().lstrip(".")

            import pandas as pd
            df = pd.DataFrame(records)

            if ext == "csv":
                df.to_csv(path, index=False)
            elif ext == "json":
                import json
                path.write_text(json.dumps(records, indent=4))
            elif ext in {"xlsx", "xls"}:
                df.to_excel(path, index=False)
            else:
                raise ValueError("Unsupported file extension. Use .csv, .json, or .xlsx.")

        return result

    # ------------------------------------------------------------------
    # Solver convenience API
    # ------------------------------------------------------------------

    def solve(self, *args, **kwargs):
        """Solve this network with ``SteadyState``.

        This is equivalent to ``SteadyState(network).solve(...)`` and keeps the
        solver available from the network for a simpler user-facing API.
        """
        from fullflow.Solvers import SteadyState

        return SteadyState(self).solve(*args, **kwargs)

    def static_evaluate(self, *args, **kwargs):
        """Evaluate this network without nonlinear solving.

        Equivalent to ``SteadyState(network).static_evaluate(...)``.
        """
        from fullflow.Solvers import SteadyState

        return SteadyState(self).static_evaluate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = [f"Network: {self.name}", f"Components ({len(self.component_list)}):"]
        lines.extend(
            f"  ├─ [{component.__class__.__name__}]: {component.name}"
            for component in self.component_list
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Network(name={self.name!r}, components={len(self.component_list)})"
