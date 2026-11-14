from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any

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

    @staticmethod
    def _component_references(value: Any, seen: set[int] | None = None):
        if isinstance(value, Component):
            yield value
            return
        if seen is None:
            seen = set()
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, State):
            if value._expr is None:
                yield from Component._component_references(value._value, seen)
        elif isinstance(value, Mapping):
            for key, item in tuple(value.items()):
                yield from Component._component_references(key, seen)
                yield from Component._component_references(item, seen)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in tuple(value):
                yield from Component._component_references(item, seen)

    def _validate_component_reference(
        self, reference: "Component", attribute: str, registered_ids: set[int] | None = None,
    ) -> None:
        if reference.network is not self.network:
            raise ValueError(
                f"{self.name}.{attribute} references {reference.name!r} from another Network. "
                "Referenced Components must belong to the same Network."
            )
        registered = (
            id(reference) in registered_ids
            if registered_ids is not None
            else any(item is reference for item in self.network.component_list)
        )
        if not registered:
            raise ValueError(
                f"{self.name}.{attribute} references {reference.name!r}, "
                "which is no longer registered in this Network."
            )

    def initialize_attribute(
        self,
        value: Any = None,
        is_default_value: bool = False,
    ) -> Any:
        """Convert numerical inputs to States while retaining Component references."""
        references = tuple(self._component_references(value))
        for reference in references:
            self._validate_component_reference(reference, "constructor input")

        if isinstance(value, Component) or (
            references and isinstance(value, (Mapping, list, tuple, set, frozenset))
        ):
            return value
        if isinstance(value, State) or getattr(type(value), "_fullflow_state_array", False):
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
    def dt(self) -> float:
        """Most recent transient timestep supplied by the solver."""
        return self._transient_dt

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
        if isinstance(value, Component):
            return f"<{type(value).__name__}: {value.name}>"
        if isinstance(value, State):
            if not value.is_assigned:
                return "<uninitialized>"
            return Component._format_value(value.value)

        if isinstance(value, list):
            return [Component._format_value(item) for item in tuple(value)]

        if isinstance(value, tuple):
            return tuple(Component._format_value(item) for item in value)

        if isinstance(value, dict):
            return {
                key: Component._format_value(item)
                for key, item in tuple(value.items())
            }

        return value

    def __str__(self) -> str:
        skip_attrs = {"network"} | self.ignored_export_attributes
        lines = [f"Component {self.name} ({self.__class__.__name__})"]

        for attr, value in tuple(self.__dict__.items()):
            if attr in skip_attrs:
                continue
            lines.append(f"    {attr}: {self._format_value(value)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
