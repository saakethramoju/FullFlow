from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .Composition import Composition
from .State import is_state_like

if TYPE_CHECKING:
    from fullflow.System import Balance, Component, State


class Network:
    def __init__(self, name: str) -> None:
        self.name = name
        self.component_list: list[Component] = []
        self.balance_list: list[Balance] = []
        self.model_list: list[Any] = []
        self.tracked_state_list: list[tuple[str, State]] = []
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def mark_structure_changed(self) -> None:
        self._version += 1

    def _add_unique(self, item: Any, item_list: list[Any]) -> None:
        if item not in item_list:
            item_list.append(item)
            self.mark_structure_changed()

    def add_component(self, component: Component) -> None:
        self._add_unique(component, self.component_list)

    def remove_component(self, component: Component) -> None:
        if component not in self.component_list:
            raise ValueError(
                f"Component {component.name!r} is not registered with network {self.name!r}."
            )
        self.component_list.remove(component)
        self.mark_structure_changed()

    def add_balance(self, balance: Balance) -> None:
        self._add_unique(balance, self.balance_list)

    def add_model(self, model: Any) -> None:
        self._add_unique(model, self.model_list)

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


    @staticmethod
    def _collect_iteration_variables(owners: list[Any]) -> list[State]:
        variables: list[State] = []
        for owner in owners:
            variables.extend(owner.iteration_variables)
        return variables

    @property
    def component_iteration_variables(self) -> list[State]:
        return self._collect_iteration_variables(self.component_list)

    @property
    def balance_iteration_variables(self) -> list[State]:
        return self._collect_iteration_variables(self.balance_list)

    @property
    def iteration_variable_states(self) -> list[State]:
        return self.component_iteration_variables + self.balance_iteration_variables

    @staticmethod
    def _state_label(owner: Any, state: State) -> str:
        for attr_name, attr_value in owner.__dict__.items():
            if attr_value is state:
                return f"{owner.name}:{attr_name}"

            if isinstance(attr_value, Composition):
                for species, species_state in attr_value.fraction.items():
                    if species_state is state:
                        return f"{owner.name}:{attr_name}.{species}"

        return f"{owner.name}:<unknown>"

    @property
    def iteration_variable_labels(self) -> list[str]:
        labels: list[str] = []
        for owner in self.component_list + self.balance_list:
            for state in owner.iteration_variables:
                labels.append(self._state_label(owner, state))
        return labels

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if is_state_like(value):
            if not value.is_assigned:
                return "<uninitialized>"
            try:
                return Network._safe_value(value.value)
            except Exception:
                return "<unavailable>"

        if isinstance(value, list):
            return [Network._safe_value(item) for item in value]

        if isinstance(value, tuple):
            return tuple(Network._safe_value(item) for item in value)

        if isinstance(value, dict):
            return {
                key: Network._safe_value(item)
                for key, item in value.items()
            }

        return value

    @staticmethod
    def _append_record(
        records: list[dict[str, Any]],
        owner_name: str,
        owner_type: str,
        attribute: str,
        value: Any,
    ) -> None:
        records.append(
            {
                "component_name": owner_name,
                "component_type": owner_type,
                "attribute": attribute,
                "value": value,
            }
        )

    def _export_owner(
        self,
        owner: Any,
        records: list[dict[str, Any]],
        ignored_attributes: set[str],
    ) -> None:
        for attr_name, attr_value in owner.__dict__.items():
            if attr_name in ignored_attributes or attr_name.startswith("_"):
                continue

            if isinstance(attr_value, Composition):
                if not attr_value.is_assigned:
                    self._append_record(
                        records,
                        owner.name,
                        owner.__class__.__name__,
                        attr_name,
                        "<uninitialized>",
                    )
                    continue

                for species, state in attr_value.fraction.items():
                    self._append_record(
                        records,
                        owner.name,
                        owner.__class__.__name__,
                        f"{attr_name}.{species}",
                        self._safe_value(state),
                    )
                continue

            self._append_record(
                records,
                owner.name,
                owner.__class__.__name__,
                attr_name,
                self._safe_value(attr_value),
            )

    def save(self, filename: str | None = None, return_type: str = "dict"):
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

        dataframe = None
        if return_type == "dict":
            result = records
        elif return_type == "dataframe":
            import pandas as pd
            dataframe = pd.DataFrame(records)
            result = dataframe
        else:
            raise ValueError("return_type must be 'dict' or 'dataframe'.")

        if filename is not None:
            path = Path(filename)
            extension = path.suffix.lower().lstrip(".")

            if extension == "json":
                import json
                path.write_text(json.dumps(records, indent=4))
            elif extension in {"csv", "xlsx", "xls"}:
                if dataframe is None:
                    import pandas as pd
                    dataframe = pd.DataFrame(records)
                if extension == "csv":
                    dataframe.to_csv(path, index=False)
                else:
                    dataframe.to_excel(path, index=False)
            else:
                raise ValueError("Unsupported file extension. Use .csv, .json, or .xlsx.")

        return result

    def __str__(self) -> str:
        lines = [
            f"Network: {self.name}",
            f"Components ({len(self.component_list)}):",
        ]
        lines.extend(
            f"  ├─ [{component.__class__.__name__}]: {component.name}"
            for component in self.component_list
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Network(name={self.name!r}, components={len(self.component_list)})"
