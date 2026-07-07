from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

from .Model import ModelOption
from .State import State, label_state_refs

if TYPE_CHECKING:
    from fullflow.System import Network


class Component:
    # Components normally participate in both steady-state and transient
    # solves.  Command/procedure/controller components can set this to True
    # so steady-state solves leave their outputs at the current value while
    # transient solves evaluate them normally.
    """Base class for all physical, empirical, control, and helper components.

        A component owns a named piece of model physics and registers itself with a
        ``Network``.  Subclasses usually implement ``evaluate_states()`` and, when
        necessary, expose ``balances`` and/or ``dynamics`` properties.  FullFlow
        deliberately keeps the component contract simple so users can write custom
        components with normal Python classes instead of a special equation DSL.

        Component authoring pattern
        ---------------------------
        ``__init__`` should accept ``name`` and ``network`` first, then physical
        inputs and optional output ``State`` objects.  The constructor should call
        ``self.setup()``.  The setup method inspects the constructor signature,
        converts plain values to ``State`` objects, preserves supplied state-like
        objects, creates optional output states for ``None`` arguments, and registers
        the component with the network.

        Solver hooks
        ------------
        ``evaluate_states()`` computes outputs and residual values from current
        inputs.  ``balances`` returns ``[(variable, residual), ...]`` for algebraic
        closure equations.  ``dynamics`` returns either ``[(state, derivative), ...]``
        for directly integrated states or ``[(iteration_state, stored_state,
        derivative), ...]`` when a convenient variable is iterated while a different
        conserved quantity is integrated.  ``pre_evaluation()`` is called before
        solver residual collection and is useful for lookups or command schedules."""
    TRANSIENT_ONLY = False

    _iteration_variable_names: tuple[str, ...] = ()
    _fullflow_setup_cache: tuple[tuple[str, ...], dict[str, Any]] | None = None

    def __init__(self, name: str, network: Network) -> None:
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
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
        """Initialize a subclass instance from its constructor arguments.

        ``setup`` must be called directly from a component constructor that has
        ``name`` and ``network`` local variables.  It reads the subclass
        constructor signature, initializes the component identity, converts
        declared physical inputs and optional outputs into FullFlow-compatible
        attributes, labels nested states for diagnostics, and registers the
        component with its network.

        User-written components normally do not need to override this method.
        They should call ``self.setup()`` at the end of ``__init__`` after any
        custom pre-setup flags have been assigned.
        """
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
        """Convert one constructor argument into the stored component attribute.

        Existing ``State`` objects are preserved.  Objects exposing an
        ``as_state`` method, such as lookup attributes, are converted through
        that method.  All other values are wrapped in a new ``State`` so
        component equations can consistently read ``.value`` regardless of
        whether the user passed a scalar constant or an explicit state object.
        """
        if isinstance(value, State):
            return value

        as_state = getattr(type(value), "as_state", None)

        if callable(as_state):
            return as_state(value)

        return State(value)

    def initialize_component(self, name: str, network: Network) -> None:
        """Store component identity and register the component with a network.

        This method is called by ``setup`` after validating that the constructor
        supplied a name and network.  It also initializes transient timestep
        bookkeeping used by components that need ``set_transient_context``.
        """
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
        """Create a reusable model-option template for this component class.

        Use ``ComponentClass.model(...)`` inside a ``Model`` definition when a
        solver should be able to build this component as one selectable option.
        The returned object stores constructor keyword arguments but does not
        build the component until the model option is selected by a solver.
        """
        cls._validate_template_arguments("model", name, kwargs)

        return ModelOption(
            name or cls.__name__,
            component_class=cls,
            kwargs=kwargs,
            component_name=name,
        )

    @classmethod
    def template(cls, name: str | None = None, **kwargs: Any) -> ModelOption:
        """Alias for :meth:`model` kept for readable model-option declarations.

        ``template`` and ``model`` return the same ``ModelOption`` object.
        """
        cls._validate_template_arguments("template", name, kwargs)
        return cls.model(name, **kwargs)

    def active_in_solver(self, solve_mode: str) -> bool:
        """Return whether this component should run in the requested solver.

        ``TRANSIENT_ONLY`` is intentionally simple for user-written components:
        set ``TRANSIENT_ONLY = True`` on a command, sequence, or controller and
        steady-state solves will skip its hooks/equations without resetting any
        of its States.  Transient solves still evaluate it normally.
        """
        if bool(getattr(self, "TRANSIENT_ONLY", False)):
            return solve_mode == "transient"
        return True

    def pre_evaluation(self) -> None:
        """Run pre-residual bookkeeping before component evaluation.
        
                Solvers call this hook before ordinary ``evaluate_states()`` passes.  It
                is used by lookups, schedules, and instrumentation components that need
                to update inputs before residual equations are collected."""
        pass

    def evaluate_states(self) -> None:
        """Update the component outputs from the current input States.

        Most custom components only need this method.  Use it to calculate any
        output States and any named derivative/error quantities used by
        ``dynamics`` or ``balances``.
        """
        pass

    @property
    def dynamics(self) -> list[tuple]:
        """Dynamic equations owned by this component.

        Use ``dynamics`` for real storage, inertia, or capacitance.  A dynamic
        equation is either::

            (state, derivative)

        or, for conservative variables solved through convenient thermodynamic
        variables::

            (solve_variable, integrated_state, derivative)

        Steady-state solves drive each derivative to zero.  Transient solves
        integrate each derivative.
        """
        return []

    @property
    def balances(self) -> list[tuple]:
        """Algebraic equations owned by this component.

        Use ``balances`` only for equations that do **not** represent a time
        derivative.  Each entry is::

            (variable_to_solve, residual_that_should_be_zero)

        Examples include a pump pressure match, a map inversion, or a controller
        target.  Do not put conservation derivatives here; those belong in
        ``dynamics``.
        """
        return []

    def set_transient_context(self, *, dt: float) -> None:
        """Receive timestep context from the transient solver.

        The base implementation stores the trial timestep in ``_transient_dt``.
        Components with rate limits, commands, or time-discrete behavior can
        override this method, but should usually call ``super()`` first.
        """
        self._transient_dt = float(dt)

    @property
    def ignored_export_attributes(self) -> set[str]:
        """Attribute names to omit from HDF5/record exports for this component.

        Components can override this property to suppress large arrays, cached
        helper objects, raw map tables, or internal bookkeeping that should not
        appear in normal solution records.
        """
        return set()

    @property
    def export_attributes(self) -> dict[str, Any]:
        """Additional attributes to include in solution exports.

        Components can override this when the exported HDF5 names should be
        cleaner than the internal runtime attributes.  The generic exporter
        treats these exactly like normal component attributes, without requiring
        the component to store duplicate State objects on ``__dict__``.
        """
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
