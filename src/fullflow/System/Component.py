from __future__ import annotations

import inspect
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any

from .Model import ModelOption
from .State import State, label_state_refs

if TYPE_CHECKING:
    from fullflow.System import Network


class Component:
    # Transient-only components retain their current output during steady state.
    """Base class for all physical, empirical, control, and helper components."""
    TRANSIENT_ONLY = False

    _iteration_variable_names: tuple[str, ...] = ()
    _fullflow_setup_cache: tuple[tuple[str, ...], dict[str, Any]] | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        constructor = cls.__dict__.get("__init__")
        if constructor is None:
            return

        @wraps(constructor)
        def registered_init(self, *args, **kwargs):
            try:
                constructor(self, *args, **kwargs)
            except Exception:
                network = self.__dict__.get("network")
                if network is not None and self in network.component_list:
                    network.remove_component(self)
                raise

        cls.__init__ = registered_init

    def __init__(self, name: str, network: Network) -> None:
        """Initialize the object and register any FullFlow state wiring."""
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
        """Initialize a subclass instance from its constructor arguments."""
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
            attribute = self.initialize_attribute(value, is_default_value)
            label_state_refs(attribute, f"{self.name}:{attr_name}")
            setattr(self, attr_name, attribute)

    def initialize_attribute(
        self,
        value: State | float | int | str | bool | None = None,
        is_default_value: bool = False,
    ) -> Any:
        """Convert one constructor argument into the stored component attribute."""
        if isinstance(value, State):
            return value

        as_state = getattr(type(value), "as_state", None)

        if callable(as_state):
            return as_state(value)

        return State(value)

    def initialize_component(self, name: str, network: Network) -> None:
        """Store component identity and register the component with a network."""
        self.name = name
        self.network = network
        self._transient_dt = 0.0
        self.network.add_component(self)

    @staticmethod
    def _looks_like_network(value: Any) -> bool:
        return (
            value is not None
            and not isinstance(value, str)
            and callable(getattr(value, "add_component", None))
            and callable(getattr(value, "add_model", None))
        )

    @classmethod
    def _validate_template_arguments(
        cls,
        method_name: str,
        name: str | None,
        kwargs: dict[str, Any],
    ) -> None:
        if name is not None and not isinstance(name, str):
            if cls._looks_like_network(name):
                raise TypeError(
                    f"{cls.__name__}.{method_name}(...) does not accept a Network "
                    "object. Put the network on Model(...), then call "
                    f"{cls.__name__}.{method_name}(...) without the network."
                )

            raise TypeError(
                f"{cls.__name__}.{method_name}(...) first positional argument "
                f"must be a component name string or omitted. Got {type(name).__name__}."
            )

        if "network" in kwargs:
            raise TypeError(
                f"{cls.__name__}.{method_name}(...) does not accept network=.... "
                "Put the network on Model(...)."
            )

    @classmethod
    def model(cls, name: str | None = None, **kwargs: Any) -> ModelOption:
        """Create a reusable model-option template for this component class."""
        cls._validate_template_arguments("model", name, kwargs)

        return ModelOption(
            name or cls.__name__,
            component_class=cls,
            kwargs=kwargs,
            component_name=name,
        )

    @classmethod
    def template(cls, name: str | None = None, **kwargs: Any) -> ModelOption:
        """Alias for :meth:`model` kept for readable model-option declarations."""
        cls._validate_template_arguments("template", name, kwargs)
        return cls.model(name, **kwargs)

    def active_in_solver(self, solve_mode: str) -> bool:
        """Return whether this component should run in the requested solver."""
        if bool(getattr(self, "TRANSIENT_ONLY", False)):
            return solve_mode == "transient"
        return True

    def pre_evaluation(self) -> None:
        """Run pre-residual bookkeeping before component evaluation."""
        pass

    def evaluate_states(self) -> None:
        """Update the component outputs from the current input States."""
        pass

    @property
    def dynamics(self) -> list[tuple]:
        """Dynamic equations owned by this component."""
        return []

    @property
    def balances(self) -> list[tuple]:
        """Algebraic equations owned by this component."""
        return []

    def set_transient_context(self, *, dt: float) -> None:
        """Receive timestep context from the transient solver."""
        self._transient_dt = float(dt)

    @property
    def ignored_export_attributes(self) -> set[str]:
        """Attribute names to omit from HDF5/record exports for this component."""
        return set()

    @property
    def export_attributes(self) -> dict[str, Any]:
        """Additional attributes to include in solution exports."""
        return {}

    @staticmethod
    def _format_value(value: Any) -> Any:
        if isinstance(value, State):
            if not value.is_assigned:
                return "<uninitialized>"
            return Component._format_value(value.value)

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
