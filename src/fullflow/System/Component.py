from __future__ import annotations

import inspect
import sys
from numbers import Real
from typing import TYPE_CHECKING, Any

from .Composition import Composition
from .Model import ModelOption
from .State import State
from .protocols import is_state_like, resolve_value

if TYPE_CHECKING:
    from fullflow.System import Network


class Component:
    """
    Base class for FullFlow components.

    Subclasses normally assign their constructor arguments by calling
    ``self.setup()``. Numeric arguments become ``State`` objects, existing
    ``State`` and ``Composition`` objects are preserved, and the component is
    registered with its network.
    """

    _iteration_variable_names: tuple[str, ...] = ()
    _fullflow_setup_cache: tuple[tuple[str, ...], dict[str, Any]] | None = None

    def __init__(self, name: str, network: Network) -> None:
        self.setup()

    @classmethod
    def _setup_parameters(cls) -> tuple[tuple[str, ...], dict[str, Any]]:
        """Return cached constructor parameter names and defaults."""
        cache = cls.__dict__.get("_fullflow_setup_cache")
        if cache is not None:
            return cache

        signature = inspect.signature(cls.__init__)
        accepted_kinds = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }

        names: list[str] = []
        defaults: dict[str, Any] = {}

        for name, parameter in signature.parameters.items():
            if name in {"self", "name", "network"}:
                continue
            if parameter.kind not in accepted_kinds:
                continue
            names.append(name)
            if parameter.default is not inspect.Parameter.empty:
                defaults[name] = parameter.default

        cache = (tuple(names), defaults)
        cls._fullflow_setup_cache = cache
        return cache

    def setup(self) -> None:
        """Initialize component attributes from the caller's constructor."""
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
    ):
        """Normalize one constructor argument into the stored attribute value."""
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


    # ------------------------------------------------------------------
    # Convenience helpers for custom components
    # ------------------------------------------------------------------

    @staticmethod
    def state_value(value: Any) -> Any:
        """Return ``value.value`` for State-like objects, otherwise ``value``.

        This keeps custom component equations readable without forcing users to
        write repetitive ``x.value`` plumbing everywhere.
        """
        return resolve_value(value)

    @staticmethod
    def value(value: Any) -> Any:
        """Short alias for :meth:`state_value` for user-written components."""
        return resolve_value(value)

    @staticmethod
    def numeric(value: Any) -> float:
        """Return a numeric ``float`` from a State-like or numeric input."""
        resolved = resolve_value(value)
        return float(resolved)

    @staticmethod
    def is_state_like(value: Any) -> bool:
        """Return True for State and State-compatible proxy objects."""
        return is_state_like(value)

    def values(self, *attribute_names: str) -> tuple[Any, ...]:
        """Return current values for named component attributes."""
        return tuple(
            self.state_value(getattr(self, attribute_name))
            for attribute_name in attribute_names
        )

    def assign_state(self, attribute_name: str, value: Any) -> None:
        """Assign a value to a named State-like output attribute."""
        getattr(self, attribute_name).value = value

    def assign(self, attribute_name: str, value: Any) -> None:
        """Short alias for :meth:`assign_state`."""
        self.assign_state(attribute_name, value)

    def make_state(self, value: Any = None, **kwargs: Any) -> State:
        """Create a ``State`` from user code without importing ``State``."""
        return State(value, **kwargs)

    def residual(self, expression: Any, scale: float | None = None) -> float:
        """Return a scalar residual, optionally normalized by ``scale``.

        This is a convenience for custom components; existing components can
        still return raw numbers or States from their ``residuals`` property.
        """
        value = float(resolve_value(expression))
        if scale is None:
            return value
        scale = float(scale)
        if scale == 0.0:
            return value
        return value / scale

    def iteration_states(self, *attribute_names: str) -> list[State]:
        """Build an ``iteration_variables`` list from attribute names."""
        return [getattr(self, attribute_name) for attribute_name in attribute_names]

    @classmethod
    def model(cls, name: str | None = None, **kwargs) -> ModelOption:
        """Create a deferred ``ModelOption`` for this component class."""
        return ModelOption(
            name or cls.__name__,
            component_class=cls,
            kwargs=kwargs,
        )

    def pre_evaluation(self) -> None:
        pass

    @property
    def iteration_variables(self) -> list[State]:
        return self.iteration_states(*self._iteration_variable_names)

    def evaluate_states(self) -> None:
        pass

    @property
    def residuals(self) -> list[float]:
        return []

    @property
    def ignored_export_attributes(self) -> set[str]:
        return set()

    @staticmethod
    def _format_value(value):
        if isinstance(value, State):
            if not value.is_assigned:
                return "<uninitialized>"
            try:
                return value.value
            except Exception:
                return "<unavailable>"

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
