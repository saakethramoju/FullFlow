class ModelOption:
    """Deferred component option used by `Model`."""
    def __init__(
        self,
        name: str,
        *model_options: "ModelOption",
        component_class: type | None = None,
        kwargs: dict | None = None,
        components: list["ModelOption"] | None = None,
    ):
        self.name = name
        self.component_class = component_class
        self.kwargs = kwargs or {}

        # Allow grouped options to be passed positionally or by keyword.
        self.components = list(model_options)

        if components is not None:
            self.components.extend(components)

        self._validate()

    def _validate(self) -> None:
        """
        Validate whether this option is a single component or a component group.
        """

        is_group = len(self.components) > 0
        is_single = self.component_class is not None

        if is_group and is_single:
            raise ValueError(
                f"{self.name}: ModelOption cannot define both "
                "component_class and grouped components."
            )

        if not is_group and not is_single:
            raise ValueError(
                f"{self.name}: ModelOption requires either "
                "component_class or grouped components."
            )

    def build(
        self,
        component_name: str,
        network,
    ):
        """
        Build this option into the network.

        Single-component options return one component.
        Group options return a list of components.
        """

        if self.is_group:
            return [
                option.build(option.name, network)
                for option in self.components
            ]

        return self.component_class(
            component_name,
            network,
            **self.kwargs,
        )

    @property
    def is_group(self) -> bool:
        """
        True if this option builds multiple components.
        """

        return len(self.components) > 0

    @property
    def component_name(self) -> str:
        """
        Name of the component class or component group.
        """

        if self.is_group:
            return "Group"

        return self.component_class.__name__

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.is_group:
            return (
                f"ModelOption("
                f"name={self.name!r}, "
                f"components={[component.name for component in self.components]})"
            )

        return (
            f"ModelOption("
            f"name={self.name!r}, "
            f"component_class={self.component_class.__name__})"
        )



class Model:
    """Collection of alternative component implementations."""
    def __init__(
        self,
        name: str,
        network,
        *model_options: ModelOption,
        components: list[ModelOption] | None = None,
        order: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        name:
            Name assigned to the model.

        network:
            Network the selected option will be added to.

        *model_options:
            ModelOptions passed positionally.

        components:
            Optional list of ModelOptions. Kept for compatibility with the
            earlier components=[...] API.

        order:
            Optional list of option names defining the try order.
            Defaults to declaration order.
        """

        self.name = name
        self.network = network

        # Support both positional options and components=[...] options.
        option_list = list(model_options)

        if components is not None:
            option_list.extend(components)

        self._option_list = option_list

        # Store options by user-facing option name.
        self.components = {
            component.name: component
            for component in option_list
        }

        # Default to the same order the options were provided in.
        self.order = (
            order
            if order is not None
            else [component.name for component in option_list]
        )

        self._validate()

        self.active_option_name = None
        self.active_component = None

        # Register this model with the network so solvers can find it.
        self.network.add_model(self)

    def _validate(self) -> None:
        """
        Validate component names and order entries.
        """

        if len(self._option_list) == 0:
            raise ValueError(f"{self.name}: Model requires at least one option.")

        if len(self.components) != len(self._option_list):
            raise ValueError(
                f"{self.name}: duplicate model option names found. "
                f"Option names must be unique."
            )

        invalid_options = [
            option_name
            for option_name in self.order
            if option_name not in self.components
        ]

        if invalid_options:
            raise ValueError(
                f"{self.name}: order contains invalid options "
                f"{invalid_options}. Valid options are "
                f"{list(self.components)}."
            )

    def build(
        self,
        option_name: str | None = None,
    ):
        """
        Build and return the selected option.

        If no option name is supplied, the first option in self.order is used.
        """

        if self.active_component is not None:
            raise RuntimeError(
                f"{self.name}: model has already built "
                f"option {self.active_option_name!r}."
            )

        option_name = option_name or self.order[0]

        if option_name not in self.components:
            raise ValueError(
                f"{self.name}: unknown model option {option_name!r}. "
                f"Valid options are {list(self.components)}."
            )

        option = self.components[option_name]

        self.active_option_name = option_name
        self.active_component = option.build(self.name, self.network)

        return self.active_component

    def clear(self) -> None:
        """
        Remove the active component or component group from the network.
        """

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

    def replace(
        self,
        option_name: str,
    ):
        """
        Replace the active option with another option.

        If no option is active yet, this simply builds the requested option.
        """

        self.clear()
        return self.build(option_name)


    def next(self) -> str:
        """
        Return the name of the next model option.

        This does not build the next option.
        """

        if self.active_option_name is None:
            return self.order[0]

        current_index = self.order.index(self.active_option_name)
        next_index = current_index + 1

        if next_index >= len(self.order):
            raise RuntimeError(f"{self.name}: no remaining model options.")

        return self.order[next_index]

    def build_next(self):
        """
        Replace the active option with the next option in the order list.
        """

        return self.replace(self.next())

    @property
    def active_option(self):
        """
        Currently selected ModelOption.
        """

        if self.active_option_name is None:
            return None

        return self.components[self.active_option_name]

    @property
    def available_options(self) -> list[str]:
        """
        Available option names.
        """

        return list(self.components)

    @property
    def has_next(self) -> bool:
        """
        True if another model option remains.
        """

        if self.active_option_name is None:
            return len(self.order) > 0

        current_index = self.order.index(self.active_option_name)

        return current_index < len(self.order) - 1

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"Model("
            f"name={self.name!r}, "
            f"options={list(self.components)}, "
            f"order={self.order}, "
            f"active={self.active_option_name!r})"
        )
