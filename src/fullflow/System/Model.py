from __future__ import annotations

from typing import Any


def _looks_like_network(value: Any) -> bool:
    return (
        value is not None
        and not isinstance(value, str)
        and callable(getattr(value, "add_component", None))
        and callable(getattr(value, "add_model", None))
    )


class ModelOption:
    """Deferred component option used by ``Model``.

    A ``ModelOption`` is normally created through ``Component.template(...)``
    or ``Component.model(...)``. It can represent either one deferred component
    or a group of deferred components.
    """

    def __init__(
        self,
        name: str,
        *options: "ModelOption",
        component_class: type | None = None,
        kwargs: dict[str, Any] | None = None,
        component_name: str | None = None,
    ) -> None:
        self.name = name
        self.component_class = component_class
        self.kwargs = kwargs or {}
        self.options = list(options)
        self.component_name = component_name
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                f"ModelOption name must be a string. Got {type(self.name).__name__}."
            )

        if self.component_name is not None and not isinstance(self.component_name, str):
            if _looks_like_network(self.component_name):
                raise TypeError(
                    f"{self.name}: component_name received a Network object. "
                    "Do not pass a network to Component.template(...) or "
                    "Component.model(...). The parent Model supplies the network."
                )

            raise TypeError(
                f"{self.name}: component_name must be a string or None. "
                f"Got {type(self.component_name).__name__}."
            )

        if "network" in self.kwargs:
            raise TypeError(
                f"{self.name}: ModelOption kwargs cannot include 'network'. "
                "Do not pass network=... to Component.template(...) or "
                "Component.model(...). The parent Model supplies the network."
            )

        if "name" in self.kwargs:
            raise TypeError(
                f"{self.name}: ModelOption kwargs cannot include 'name'. "
                "Pass the component name as the first argument to "
                "Component.template(...) instead."
            )

        is_group = bool(self.options)
        is_single = self.component_class is not None

        if is_group == is_single:
            raise ValueError(
                f"{self.name}: ModelOption requires exactly one of "
                "component_class or grouped options."
            )

        for option in self.options:
            if not isinstance(option, ModelOption):
                raise TypeError(
                    f"{self.name}: grouped options must be ModelOption objects. "
                    f"Got {type(option).__name__}. Use Component.template(...) "
                    "or Component.model(...)."
                )

    @property
    def is_group(self) -> bool:
        return bool(self.options)

    @property
    def components(self) -> list["ModelOption"]:
        """Backward-compatible alias for grouped child options."""
        return self.options

    @property
    def component_type_name(self) -> str:
        return "Group" if self.is_group else self.component_class.__name__

    @property
    def component_name_or_default(self) -> str:
        return self.component_name or self.name

    def renamed(self, name: str) -> "ModelOption":
        """Return the same deferred option under a different option name."""
        if self.is_group:
            return ModelOption(name, *self.options)

        return ModelOption(
            name,
            component_class=self.component_class,
            kwargs=dict(self.kwargs),
            component_name=self.component_name,
        )

    def build(self, component_name: str, network):
        if self.is_group:
            components = []

            for option in self.options:
                built = option.build(option.component_name_or_default, network)

                if isinstance(built, list):
                    components.extend(built)
                else:
                    components.append(built)

            return components

        name = self.component_name or component_name
        return self.component_class(name, network, **self.kwargs)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.is_group:
            members = [option.name for option in self.options]
            return f"ModelOption(name={self.name!r}, options={members})"

        return (
            f"ModelOption(name={self.name!r}, "
            f"component_class={self.component_class.__name__})"
        )


class Model:
    """Collection of switchable model options.

    Preferred interface:

        model = Model("Nozzle", network, order=["isentropic", "shock"])
        model.option("isentropic", IsentropicNozzle.template(...))
        model.option("shock", IsentropicNozzle.template(..., normal_shock=True))

    If ``order`` is omitted, options are evaluated in the order they are added:

        model = Model("Nozzle", network)
        model.option("isentropic", IsentropicNozzle.template(...))
        model.option("shock", IsentropicNozzle.template(..., normal_shock=True))

    Each option may contain one deferred component or several deferred
    components. Options are built only when a solver selects them.
    """

    def __init__(
        self,
        name: str,
        network,
        *options: ModelOption,
        order: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.network = network
        self.options: dict[str, ModelOption] = {}
        self.order: list[str] = list(order) if order is not None else []
        self._explicit_order = order is not None

        self.active_option_name = None
        self.active_component = None

        for option in options:
            self.add_option(option)

        self.network.add_model(self)

    def add_option(self, option: ModelOption) -> ModelOption:
        if not isinstance(option, ModelOption):
            raise TypeError(
                f"{self.name}: options must be ModelOption objects. "
                f"Got {type(option).__name__}."
            )

        if option.name in self.options:
            raise ValueError(
                f"{self.name}: duplicate model option name {option.name!r}. "
                "Option names must be unique."
            )

        self.options[option.name] = option

        if not self._explicit_order:
            self.order.append(option.name)

        return option

    def option(self, name: str, *options: ModelOption) -> ModelOption:
        """Add a model option.

        Pass one deferred component for a single-component option, or several
        deferred components for a grouped option.
        """
        if not options:
            raise ValueError(
                f"{self.name}: option {name!r} requires at least one component template."
            )

        for option in options:
            if not isinstance(option, ModelOption):
                raise TypeError(
                    f"{self.name}: option {name!r} entries must be ModelOption objects. "
                    f"Got {type(option).__name__}. Use Component.template(...) "
                    "or Component.model(...)."
                )

        if len(options) == 1:
            option = options[0].renamed(name)
        else:
            option = ModelOption(name, *options)

        return self.add_option(option)

    def _validate_ready(self) -> None:
        if not self.options:
            raise ValueError(f"{self.name}: Model requires at least one option.")

        if not self.order:
            self.order = list(self.options)

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

        self._validate_ready()
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
        self._validate_ready()

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
        self._validate_ready()

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
