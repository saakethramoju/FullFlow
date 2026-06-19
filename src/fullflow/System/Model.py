from __future__ import annotations

from typing import Any


class ModelOption:
    def __init__(
        self,
        name: str,
        *model_options: "ModelOption",
        component_class: type | None = None,
        kwargs: dict[str, Any] | None = None,
        components: list["ModelOption"] | None = None,
    ) -> None:
        self.name = name
        self.component_class = component_class
        self.kwargs = kwargs or {}
        self.components = list(model_options)
        if components is not None:
            self.components.extend(components)
        self._validate()

    def _validate(self) -> None:
        is_group = bool(self.components)
        is_single = self.component_class is not None
        if is_group == is_single:
            raise ValueError(
                f"{self.name}: ModelOption requires exactly one of "
                "component_class or grouped components."
            )

    @property
    def is_group(self) -> bool:
        return bool(self.components)

    @property
    def component_name(self) -> str:
        return "Group" if self.is_group else self.component_class.__name__

    def build(self, component_name: str, network):
        if self.is_group:
            return [option.build(option.name, network) for option in self.components]
        return self.component_class(component_name, network, **self.kwargs)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.is_group:
            members = [component.name for component in self.components]
            return f"ModelOption(name={self.name!r}, components={members})"
        return (
            f"ModelOption(name={self.name!r}, "
            f"component_class={self.component_class.__name__})"
        )









class Model:
    def __init__(
        self,
        name: str,
        network,
        *options: ModelOption,
        order: list[str] | None = None,
    ) -> None:
        self.name = name
        self.network = network
        self.options = {option.name: option for option in options}
        self.order = order or [option.name for option in options]
        self._validate(options)

        self.active_option_name = None
        self.active_component = None
        self.network.add_model(self)

    def _validate(self, options: tuple[ModelOption, ...]) -> None:
        if not options:
            raise ValueError(f"{self.name}: Model requires at least one option.")

        if len(self.options) != len(options):
            raise ValueError(
                f"{self.name}: duplicate model option names found. "
                "Option names must be unique."
            )

        invalid_options = [name for name in self.order if name not in self.options]
        if invalid_options:
            raise ValueError(
                f"{self.name}: order contains invalid options {invalid_options}. "
                f"Valid options are {list(self.options)}."
            )

    def build(self, option_name: str | None = None):
        if self.active_component is not None:
            raise RuntimeError(
                f"{self.name}: model has already built option {self.active_option_name!r}."
            )

        option_name = option_name or self.order[0]
        if option_name not in self.options:
            raise ValueError(
                f"{self.name}: unknown model option {option_name!r}. "
                f"Valid options are {list(self.options)}."
            )

        self.active_option_name = option_name
        self.active_component = self.options[option_name].build(self.name, self.network)
        return self.active_component

    def clear(self) -> None:
        if self.active_component is None:
            return

        active_components = (
            self.active_component
            if isinstance(self.active_component, list)
            else [self.active_component]
        )

        for component in active_components:
            self.network.remove_component(component)

        self.active_component = None
        self.active_option_name = None

    def replace(self, option_name: str):
        self.clear()
        return self.build(option_name)

    def next(self) -> str:
        if self.active_option_name is None:
            return self.order[0]

        next_index = self.order.index(self.active_option_name) + 1
        if next_index >= len(self.order):
            raise RuntimeError(f"{self.name}: no remaining model options.")
        return self.order[next_index]

    def build_next(self):
        return self.replace(self.next())

    @property
    def active_option(self):
        if self.active_option_name is None:
            return None
        return self.options[self.active_option_name]

    @property
    def available_options(self) -> list[str]:
        return list(self.options)

    @property
    def has_next(self) -> bool:
        if self.active_option_name is None:
            return bool(self.order)
        return self.order.index(self.active_option_name) < len(self.order) - 1

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"Model(name={self.name!r}, options={list(self.options)}, "
            f"order={self.order}, active={self.active_option_name!r})"
        )