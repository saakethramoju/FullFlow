from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .State import State, is_state_like, label_state_refs
from fullflow.Exports.HDF5 import write_solution

if TYPE_CHECKING:
    from fullflow.System import Balance, Component, State


class Network:
    def __init__(self, name: str) -> None:
        self.name = name
        self.time = State(0.0)
        self.time.add_label(f"{self.name}:time")
        self.component_list: list[Component] = []
        self.balance_list: list[Balance] = []
        self.model_list: list[Any] = []
        self.tracked_state_list: list[dict[str, Any]] = []
        self.sensor_event_list: list[Any] = []
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


    def sensor_components(self) -> list[Any]:
        """Return registered Sensor-like components without touching dynamic instance attributes.

        Lookup components implement dynamic output attributes through ``__getattr__``.
        Calling ``getattr(component, "check_conditions", None)`` on every
        component can therefore accidentally ask a Lookup for an output named
        ``check_conditions``. Inspect the component type instead so only real
        class methods are considered.
        """
        return [
            component
            for component in self.component_list
            if callable(getattr(type(component), "check_conditions", None))
        ]

    def reset_sensor_events(self) -> None:
        """Clear transient sensor events before a new run."""
        self.sensor_event_list = []

    def initialize_sensor_conditions(self, time_value: float | None = None) -> None:
        """Prime Sensor condition crossing memory without logging events."""
        if time_value is None:
            time_value = float(self.time.value)

        for sensor in self.sensor_components():
            reset = getattr(type(sensor), "reset_condition_history", None)
            if callable(reset):
                reset(sensor, float(time_value))

    def check_sensor_conditions(self, time_value: float | None = None) -> list[Any]:
        """Check all Sensor condition traces and append any new events."""
        if time_value is None:
            time_value = float(self.time.value)

        events: list[Any] = []
        for sensor in self.sensor_components():
            check = getattr(type(sensor), "check_conditions", None)
            if not callable(check):
                continue
            sensor_events = list(check(sensor, float(time_value)) or [])
            events.extend(sensor_events)

        self.sensor_event_list.extend(events)
        return events

    def sensor_event_records(self) -> list[dict[str, Any]]:
        """Return sparse Sensor condition events for HDF5 export."""
        rows: list[dict[str, Any]] = []
        for event in self.sensor_event_list:
            as_record = getattr(event, "as_record", None)
            if callable(as_record):
                rows.append(dict(as_record()))
            elif isinstance(event, dict):
                rows.append(dict(event))
            else:
                rows.append({"event": str(event)})
        return rows

    def sensor_condition_trace_records(self) -> list[dict[str, Any]]:
        """Return Sensor condition trace definitions for HDF5 export."""
        rows: list[dict[str, Any]] = []
        for sensor in self.sensor_components():
            for condition in getattr(sensor, "conditions", ()) or ():
                trace = getattr(condition, "trace", None)
                if trace is None:
                    continue
                rows.append(
                    {
                        "sensor": getattr(sensor, "name", "sensor"),
                        "trace": getattr(trace, "name", "condition"),
                        "role": getattr(condition, "role", getattr(trace, "role", "")),
                        "action": getattr(condition, "action", ""),
                        "x": getattr(trace, "x", []),
                        "y": getattr(trace, "y", []),
                    }
                )
        return rows

    def sequence_components(self) -> list[Any]:
        """Return registered Sequence-like components without dynamic attribute lookup."""
        return [
            component
            for component in self.component_list
            if callable(getattr(type(component), "command_trace_records", None))
        ]

    def sequence_command_trace_records(self) -> list[dict[str, Any]]:
        """Return Sequence command trace definitions for HDF5 export."""
        rows: list[dict[str, Any]] = []
        for sequence in self.sequence_components():
            command_trace_records = getattr(type(sequence), "command_trace_records", None)
            if callable(command_trace_records):
                rows.extend(command_trace_records(sequence) or [])
        return rows

    def activate_sequence_conditions(self, events: list[Any]) -> None:
        """Activate Sequence commands and abort rules that depend on Sensor events."""
        if not events:
            return

        for sequence in self.sequence_components():
            activate = getattr(type(sequence), "activate_condition_commands", None)
            if callable(activate):
                activate(sequence, events)

            activate_abort = getattr(type(sequence), "activate_condition_aborts", None)
            if callable(activate_abort):
                activate_abort(sequence, events)

    def reset_sequence_aborts(self) -> None:
        """Reset one-run Sequence command and abort history before a transient run."""
        for sequence in self.sequence_components():
            reset_commands = getattr(type(sequence), "reset_command_history", None)
            if callable(reset_commands):
                reset_commands(sequence)

            reset_aborts = getattr(type(sequence), "reset_abort_history", None)
            if callable(reset_aborts):
                reset_aborts(sequence)

    def next_sequence_abort_time(self, current_time: float | None = None) -> float | None:
        """Return the next clean abort time requested by any Sequence."""
        if current_time is None:
            current_time = float(self.time.value)

        candidates: list[float] = []
        for sequence in self.sequence_components():
            next_abort_time = getattr(type(sequence), "next_abort_time", None)
            if not callable(next_abort_time):
                continue
            value = next_abort_time(sequence, float(current_time))
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            candidates.append(value)

        if not candidates:
            return None
        return min(candidates)

    def check_sequence_abort(self, time_value: float | None = None) -> dict[str, Any] | None:
        """Return a clean abort record if any Sequence abort rule is due."""
        if time_value is None:
            time_value = float(self.time.value)

        abort_records: list[dict[str, Any]] = []
        for sequence in self.sequence_components():
            check_abort = getattr(type(sequence), "check_abort", None)
            if not callable(check_abort):
                continue
            record = check_abort(sequence, float(time_value))
            if record:
                abort_records.append(dict(record))

        if not abort_records:
            return None
        return min(abort_records, key=lambda row: float(row.get("time", time_value)))

    def track(
        self,
        name: str,
        value: Any,
        attributes: str | list[str] | tuple[str, ...] | dict[str, str] | None = None,
        items: str | list[str] | tuple[str, ...] | dict[str, str] | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        max_items: int | None = None,
        flatten: bool = True,
    ) -> Any:
        label_state_refs(value, f"track:{name}")

        self.tracked_state_list.append(
            {
                "name": name,
                "value": value,
                "attributes": attributes,
                "items": items,
                "minimum": minimum,
                "maximum": maximum,
                "max_items": max_items,
                "flatten": flatten,
            }
        )
        self.mark_structure_changed()
        return value

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
    def _state_paths(value: Any, target: Any, prefix: str) -> list[str]:
        paths: list[str] = []
        seen: set[int] = set()

        def collect(item: Any, path: str) -> None:
            item_id = id(item)
            if item_id in seen:
                return
            seen.add(item_id)

            if item is target:
                paths.append(path)
                return

            if is_state_like(item):
                try:
                    collect(item.value, path)
                except Exception:
                    pass
                return

            if isinstance(item, dict):
                for key, child in item.items():
                    collect(child, f"{path}.{key}")
                return

            if isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    collect(child, f"{path}[{index}]")
                return

        collect(value, prefix)
        return paths

    @classmethod
    def _state_label(cls, owner: Any, state: State) -> str:
        for attr_name, attr_value in owner.__dict__.items():
            paths = cls._state_paths(attr_value, state, f"{owner.name}:{attr_name}")
            if paths:
                return paths[0]

        return f"{owner.name}:<unknown>"

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
    def _selection_map(
        selection: str | list[str] | tuple[str, ...] | dict[str, str] | None,
    ) -> dict[str, str]:
        if selection is None:
            return {}

        if isinstance(selection, str):
            return {selection: selection}

        if isinstance(selection, dict):
            return dict(selection)

        return {name: name for name in selection}

    @staticmethod
    def _object_attribute(value: Any, attribute: str) -> Any:
        current = value

        for name in attribute.split("."):
            current = getattr(current, name)

        return current

    @staticmethod
    def _filtered_items(
        values: Any,
        minimum: float | None = None,
        maximum: float | None = None,
        max_items: int | None = None,
    ) -> dict[Any, Any]:
        if isinstance(values, Mapping):
            items = dict(values)
        else:
            items = dict(values)

        if minimum is not None or maximum is not None:
            filtered_items = {}

            for key, value in items.items():
                try:
                    if minimum is not None and value < minimum:
                        continue

                    if maximum is not None and value > maximum:
                        continue
                except Exception:
                    continue

                filtered_items[key] = value

            items = filtered_items

        def sort_key(pair: tuple[Any, Any]) -> float:
            try:
                return abs(float(pair[1]))
            except Exception:
                return 0.0

        items = dict(
            sorted(
                items.items(),
                key=sort_key,
                reverse=True,
            )
        )

        if max_items is not None:
            items = dict(list(items.items())[:max_items])

        return items

    def _tracked_value(self, tracked: dict[str, Any]) -> Any:
        value = self._safe_value(tracked["value"])

        attributes = tracked["attributes"]
        items = tracked["items"]

        if attributes is None and items is None:
            return value

        data: dict[str, Any] = {}

        for label, attribute in self._selection_map(attributes).items():
            try:
                data[label] = self._safe_value(self._object_attribute(value, attribute))
            except Exception:
                data[label] = "<unavailable>"

        for label, attribute in self._selection_map(items).items():
            try:
                item_values = self._object_attribute(value, attribute)
                data[label] = self._filtered_items(
                    item_values,
                    minimum=tracked["minimum"],
                    maximum=tracked["maximum"],
                    max_items=tracked["max_items"],
                )
            except Exception:
                data[label] = "<unavailable>"

        return data

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

    def _append_tracked_records(
        self,
        records: list[dict[str, Any]],
        name: str,
        value: Any,
        flatten: bool,
        prefix: str | None = None,
    ) -> None:
        attribute = name if prefix is None else f"{prefix}.{name}"

        if flatten and isinstance(value, dict):
            for key, item in value.items():
                self._append_tracked_records(
                    records,
                    str(key),
                    item,
                    flatten=True,
                    prefix=attribute,
                )
            return

        self._append_record(
            records,
            self.name,
            "TrackedState",
            attribute,
            self._safe_value(value),
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

            self._append_record(
                records,
                owner.name,
                owner.__class__.__name__,
                attr_name,
                self._safe_value(attr_value),
            )

        # Components may expose extra export-only attributes without storing
        # duplicate runtime State objects on ``__dict__``.  Map uses this to
        # export each named input separately, for example ``chamber_pressure``
        # and ``mixture_ratio``, instead of one generic ``inputs`` dictionary.
        export_attributes = getattr(owner, "export_attributes", {})
        if callable(export_attributes):
            export_attributes = export_attributes()

        for attr_name, attr_value in dict(export_attributes).items():
            if attr_name in ignored_attributes or attr_name.startswith("_"):
                continue

            self._append_record(
                records,
                owner.name,
                owner.__class__.__name__,
                str(attr_name),
                self._safe_value(attr_value),
            )

    def tracked_records(self) -> list[dict[str, Any]]:
        """Return only user-tracked records for transient history export."""
        records: list[dict[str, Any]] = []

        for tracked in self.tracked_state_list:
            self._append_tracked_records(
                records,
                tracked["name"],
                self._tracked_value(tracked),
                flatten=tracked["flatten"],
            )

        return records

    def save(
        self,
        filename: str | None = None,
        return_type: str = "dict",
        group_path: str = "steady_state/runs/base",
        metadata: dict[str, Any] | None = None,
    ):
        return_type = return_type.lower()
        records: list[dict[str, Any]] = []

        for component in self.component_list:
            ignored = {"name", "network"} | component.ignored_export_attributes
            self._export_owner(component, records, ignored)

        for balance in self.balance_list:
            self._export_owner(balance, records, {"name", "network"})

        records.extend(self.tracked_records())

        if return_type not in {"dict", "records"}:
            raise ValueError("return_type must be 'dict' or 'records'.")

        if filename is not None:
            write_solution(
                filename,
                records,
                network_name=self.name,
                models=self.model_list,
                group_path=group_path,
                metadata=metadata,
            )

        return records

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