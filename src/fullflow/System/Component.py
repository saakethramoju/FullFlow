from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

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
        value: State | float | int | str | bool | None = None,
        is_default_value: bool = False,
    ) -> Any:
        if isinstance(value, State):
            return value

        as_state = getattr(type(value), "as_state", None)

        if callable(as_state):
            return as_state(value)

        return State(value)

    def initialize_component(self, name: str, network: Network) -> None:
        self.name = name
        self.network = network
        self._discrete_frozen = False
        self._proposed_modes = {}
        self._proposed_mode_values = {}
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
        cls._validate_template_arguments("model", name, kwargs)

        return ModelOption(
            name or cls.__name__,
            component_class=cls,
            kwargs=kwargs,
            component_name=name,
        )

    @classmethod
    def template(cls, name: str | None = None, **kwargs: Any) -> ModelOption:
        cls._validate_template_arguments("template", name, kwargs)
        return cls.model(name, **kwargs)

    def propose(
        self,
        name: str,
        value: Any,
        *,
        turn_on: float | None = None,
        turn_off: float | None = None,
        initial: bool = False,
    ) -> bool:
        """Safely propose a boolean/discrete mode owned by this component.

        Components use this for discontinuous runtime branches such as
        cavitating/noncavitating, choked/unchoked, laminar/turbulent,
        normal-shock/no-shock, limiters, or bang-bang valves.  The mode is
        stored as a normal ``State`` either by reusing an existing component
        attribute named by ``name`` or by creating a hidden internal mode State.

        In normal evaluation, the proposed mode is assigned immediately.
        During a transient nonlinear solve, the transient solver freezes
        component discrete modes before SciPy perturbs the continuous unknowns.
        While frozen, this method records the proposed value but returns the
        currently accepted value, so residual equations do not jump branches
        inside one nonlinear solve.  Proposed mode changes are committed only
        after the timestep is accepted.

        The simple form uses a boolean condition::

            is_choked = self.propose("is_choked", pressure_ratio <= critical)

        The threshold form adds hysteresis::

            is_open = self.propose(
                "is_open",
                pressure,
                turn_on=opening_pressure,
                turn_off=closing_pressure,
            )

        With hysteresis, a False mode turns True only when ``value > turn_on``;
        a True mode turns False only when ``value < turn_off``.  Between those
        thresholds, the current mode is kept.
        """
        if (turn_on is None) != (turn_off is None):
            raise ValueError(
                "turn_on and turn_off must either both be provided or both be omitted."
            )

        if name in self.__dict__:
            state = self.__dict__[name]
        else:
            state = self._proposed_modes.get(name)

            if state is None:
                state = State(bool(initial))
                self._proposed_modes[name] = state

        if not state.is_assigned:
            state.value = bool(initial)

        current_value = bool(state.value)

        if turn_on is None:
            proposed_value = bool(value)
        else:
            turn_on_value = float(turn_on)
            turn_off_value = float(turn_off)

            if turn_on_value < turn_off_value:
                raise ValueError("turn_on must be greater than or equal to turn_off.")

            signal = float(value)

            if current_value:
                proposed_value = signal > turn_off_value
            else:
                proposed_value = signal > turn_on_value

        if self._discrete_frozen:
            self._proposed_mode_values[id(state)] = (state, proposed_value)
            return current_value

        state.value = proposed_value
        self._proposed_mode_values.pop(id(state), None)
        return bool(state.value)

    def freeze_discrete(self) -> None:
        """Freeze component discrete modes before a transient nonlinear solve."""
        self._discrete_frozen = True
        self._proposed_mode_values.clear()

    def commit_discrete(self) -> None:
        """Accept proposed discrete-mode changes after an accepted timestep."""
        proposed_values = list(self._proposed_mode_values.values())
        self._discrete_frozen = False
        self._proposed_mode_values.clear()

        for state, proposed_value in proposed_values:
            state.value = proposed_value

    def reject_discrete(self) -> None:
        """Discard proposed discrete-mode changes after a failed timestep."""
        self._discrete_frozen = False
        self._proposed_mode_values.clear()

    def pre_evaluation(self) -> None:
        pass

    @property
    def iteration_variables(self) -> list[State]:
        return [getattr(self, name) for name in self._iteration_variable_names]

    def evaluate_states(self) -> None:
        pass

    @property
    def residuals(self) -> list[State | float]:
        return []

    @property
    def transient_variables(self) -> list[State]:
        return []

    @property
    def transient_states(self) -> list[State]:
        """Conserved quantities advanced by the transient integrator.

        For most dynamic components, the solver variable and the integrated
        quantity are the same State.  For example, a rotor solves for and
        integrates rotor speed.  In that common case, components only need to
        override ``transient_variables`` and ``transient_derivatives``.

        Some components should solve with one set of variables but integrate a
        different conserved quantity.  A fluid volume is the main example: it
        may solve for pressure and enthalpy while integrating mass and total
        internal energy.  Those components override this property.
        """
        return self.transient_variables

    @property
    def transient_derivatives(self) -> list[State | float]:
        return []

    @property
    def transient_algebraic_variables(self) -> list[State]:
        """Extra new-time unknowns used only during transient solves.

        These States are written by the transient nonlinear solver, but they do
        not create their own integration residuals.  They are useful for
        variable-geometry components where a Balance or another algebraic
        equation closes the extra unknown.
        """
        return []

    @property
    def transient_history_states(self) -> list[State]:
        """Extra States whose previous timestep value should be stored.

        Transient states are stored automatically.  Components can add related
        algebraic States here when they need old/new differences, such as a
        moving volume boundary used for boundary work.
        """
        return []

    def set_transient_context(self, *, dt: float) -> None:
        """Receive timestep context from the transient solver.

        Components normally do not need this.  Variable-volume energy balances
        use it to compute a backward-Euler volume derivative from current and
        previous volume values.
        """
        self._transient_dt = float(dt)

    @property
    def ignored_export_attributes(self) -> set[str]:
        return {
            "_discrete_frozen",
            "_proposed_modes",
            "_proposed_mode_values",
        }

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