from __future__ import annotations

import inspect
import sys
from numbers import Real
from typing import TYPE_CHECKING, Any

from .Composition import Composition
from .Model import ModelOption
from .State import State

if TYPE_CHECKING:
    from fullflow.System import Network


class Component:
    _iteration_variable_names: tuple[str, ...] = ()
    _fullflow_setup_cache: tuple[tuple[str, ...], dict[str, Any]] | None = None

    def __init__(self, name: str, network: Network) -> None:
        self.setup()

    @classmethod
    def _setup_parameters(cls) -> tuple[tuple[str, ...], dict[str, Any]]:
        cache = cls.__dict__.get("_fullflow_setup_cache")
        if cache is not None:
            return cache

        signature = inspect.signature(cls.__init__)
        names: list[str] = []
        defaults: dict[str, Any] = {}

        for name, parameter in signature.parameters.items():
            if name in {"self", "name", "network"}:
                continue

            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                continue

            names.append(name)
            if parameter.default is not inspect.Parameter.empty:
                defaults[name] = parameter.default

        cache = (tuple(names), defaults)
        cls._fullflow_setup_cache = cache
        return cache

    def setup(self) -> None:
        frame = sys._getframe(1)
        local_vars = frame.f_locals

        try:
            name = local_vars["name"]
            network = local_vars["network"]
        except KeyError as exc:
            raise RuntimeError(
                "Component.setup() must be called directly from a constructor "
                "with `name` and `network` parameters."
            ) from exc

        self.initialize_component(name, network)

        parameters, defaults = self._setup_parameters()
        for attr_name in parameters:
            if attr_name not in local_vars:
                continue

            value = local_vars[attr_name]
            is_default_value = (
                attr_name in defaults
                and value is defaults[attr_name]
            )
            setattr(
                self,
                attr_name,
                self.initialize_attribute(value, is_default_value),
            )

    def initialize_attribute(
        self,
        value: State | Composition | float | int | str | bool | None = None,
        is_default_value: bool = False,
    ) -> Any:
        if is_default_value and isinstance(value, Composition):
            return Composition()

        if isinstance(value, (State, Composition)):
            return value

        if value is None:
            return State()

        if isinstance(value, bool):
            return value

        if isinstance(value, Real):
            return State(float(value))

        return value

    def initialize_component(self, name: str, network: Network) -> None:
        self.name = name
        self.network = network
        self.network.add_component(self)

    @classmethod
    def model(cls, name: str | None = None, **kwargs: Any) -> ModelOption:
        return ModelOption(
            name or cls.__name__,
            component_class=cls,
            kwargs=kwargs,
        )

    def pre_evaluation(self) -> None:
        pass

    @property
    def iteration_variables(self) -> list[State]:
        return [getattr(self, name) for name in self._iteration_variable_names]

    def evaluate_states(self) -> None:
        pass

    @property
    def residuals(self) -> list[float]:
        return []

    @property
    def ignored_export_attributes(self) -> set[str]:
        return set()

    @staticmethod
    def _format_value(value: Any) -> Any:
        if isinstance(value, State):
            if not value.is_assigned:
                return "<uninitialized>"
            return value.value

        if isinstance(value, list):
            return [Component._format_value(item) for item in value]

        if isinstance(value, tuple):
            return tuple(Component._format_value(item) for item in value)

        if isinstance(value, dict):
            return {
                key: Component._format_value(item)
                for key, item in value.items()
            }

        return value

    def __str__(self) -> str:
        skip_attrs = {"network"} | self.ignored_export_attributes
        lines = [f"Component {self.name} ({self.__class__.__name__})"]

        for attr, value in self.__dict__.items():
            if attr in skip_attrs:
                continue
            lines.append(f"    {attr}: {self._format_value(value)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"