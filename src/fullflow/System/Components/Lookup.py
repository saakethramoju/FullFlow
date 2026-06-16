from __future__ import annotations

from collections import OrderedDict
import inspect
import math
import operator
from typing import Any, Callable, Generic, TypeVar, TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


T = TypeVar("T")

_SIGNATURE_CACHE: dict[int, tuple[inspect.Signature | None, set[str], bool]] = {}
_UPDATE_ACCEPTS_SOLVE_CACHE: dict[type, bool] = {}

_MISSING = object()
_UNAVAILABLE = object()


class LookupAttribute:
    """
    Dynamic state-like proxy for a Lookup input or output attribute.

    Supports:
        FuelSource.pressure
        FuelSource.density
        eq.pressure
        eq.temperature
        reactants.mixture_ratio

    Also supports FullFlow-style math:
        Chamber.mass_flow_in = FuelInjector.mass_flow + OxInjector.mass_flow
        reactants.mixture_ratio = OxInjector.mass_flow / FuelInjector.mass_flow
    """

    _fullflow_state_like = True
    _fullflow_assignable_state_like = True
    __slots__ = ("lookup", "name")

    def __init__(self, lookup: "Lookup", name: str):
        self.lookup = lookup
        self.name = name

    @property
    def is_input(self) -> bool:
        return self.lookup.input_is_active(self.name)

    @property
    def is_output(self) -> bool:
        return not self.is_input

    @property
    def value(self) -> Any:
        if self.is_input:
            return self.lookup.get_input(self.name)

        return self.lookup.get_output(self.name)

    @value.setter
    def value(self, new_value: Any) -> None:
        if self.is_input or self.lookup.accepts_input(self.name):
            self.lookup.set_input(self.name, new_value)
            return

        self.lookup.set_output_guess(self.name, new_value)

    @property
    def numeric_value(self) -> float:
        return float(self.value)

    @property
    def is_assigned(self) -> bool:
        if self.is_input:
            return self.lookup.input_is_assigned(self.name)

        return self.lookup.output_is_assigned(self.name)

    @property
    def is_numeric(self) -> bool:
        try:
            float(self.value)
            return True
        except Exception:
            return False

    @property
    def is_derived(self) -> bool:
        return True

    def _constraint_state(self) -> State | None:
        """Return the backing State that should provide solver constraints."""
        if self.name in self.lookup.kwargs:
            current = self.lookup.kwargs[self.name]

            if isinstance(current, State):
                return current

        state = self.lookup._output_states.get(self.name)

        if state is not None:
            return state

        return self.lookup._input_fallback_states.get(self.name)

    @property
    def lower_bound(self) -> float:
        state = self._constraint_state()

        if state is None:
            return -math.inf

        return state.lower_bound

    @property
    def upper_bound(self) -> float:
        state = self._constraint_state()

        if state is None:
            return math.inf

        return state.upper_bound

    @property
    def bounds(self) -> tuple[float, float]:
        state = self._constraint_state()

        if state is None:
            return self.lower_bound, self.upper_bound

        return state.bounds

    @property
    def has_bounds(self) -> bool:
        state = self._constraint_state()
        return bool(state is not None and state.has_bounds)

    @property
    def keep_feasible(self) -> bool:
        state = self._constraint_state()
        return bool(state is not None and state.keep_feasible)

    def as_state(self) -> State:
        return State._derived(lambda: self.value)

    def set(self, value: Any) -> "LookupAttribute":
        self.value = value
        return self

    @staticmethod
    def _resolve(x: Any) -> Any:
        if isinstance(x, LookupAttribute):
            return x.value

        if isinstance(x, Lookup):
            return x.value

        if isinstance(x, State):
            return x.value

        if isinstance(x, tuple):
            return tuple(LookupAttribute._resolve(v) for v in x)

        if isinstance(x, list):
            return [LookupAttribute._resolve(v) for v in x]

        if isinstance(x, dict):
            return {
                LookupAttribute._resolve(k): LookupAttribute._resolve(v)
                for k, v in x.items()
            }

        return x

    def _binary_state(self, other: Any, op: Callable[[Any, Any], Any]) -> State:
        return State._derived(lambda: op(self.value, self._resolve(other)))

    def _rbinary_state(self, other: Any, op: Callable[[Any, Any], Any]) -> State:
        return State._derived(lambda: op(self._resolve(other), self.value))

    def __add__(self, other: Any) -> State:
        return self._binary_state(other, operator.add)

    def __radd__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.add)

    def __sub__(self, other: Any) -> State:
        return self._binary_state(other, operator.sub)

    def __rsub__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.sub)

    def __mul__(self, other: Any) -> State:
        return self._binary_state(other, operator.mul)

    def __rmul__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.mul)

    def __truediv__(self, other: Any) -> State:
        return self._binary_state(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.truediv)

    def __floordiv__(self, other: Any) -> State:
        return self._binary_state(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.floordiv)

    def __mod__(self, other: Any) -> State:
        return self._binary_state(other, operator.mod)

    def __rmod__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.mod)

    def __pow__(self, other: Any) -> State:
        return self._binary_state(other, operator.pow)

    def __rpow__(self, other: Any) -> State:
        return self._rbinary_state(other, operator.pow)

    def __neg__(self) -> State:
        return State._derived(lambda: -self.value)

    def __pos__(self) -> State:
        return State._derived(lambda: +self.value)

    def __abs__(self) -> State:
        return State._derived(lambda: abs(self.value))

    def __eq__(self, other: Any) -> bool:
        return self.value == self._resolve(other)

    def __ne__(self, other: Any) -> bool:
        return self.value != self._resolve(other)

    def __lt__(self, other: Any) -> bool:
        return self.value < self._resolve(other)

    def __le__(self, other: Any) -> bool:
        return self.value <= self._resolve(other)

    def __gt__(self, other: Any) -> bool:
        return self.value > self._resolve(other)

    def __ge__(self, other: Any) -> bool:
        return self.value >= self._resolve(other)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)

    def __repr__(self) -> str:
        try:
            return repr(self.value)
        except Exception:
            return f"{self.lookup.name}.{self.name}"

    def __str__(self) -> str:
        try:
            return str(self.value)
        except Exception:
            return f"{self.lookup.name}.{self.name}"

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        current = self.value
        current[key] = value
        self.value = current

    def __iter__(self):
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, item: Any) -> bool:
        return item in self.value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        try:
            value = self.value
        except Exception as exc:
            raise AttributeError(name) from exc

        return getattr(value, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value(*args, **kwargs)

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class Lookup(Component, Generic[T]):
    """
    Wrap a function, class, or external model as a FullFlow component.

    ``Lookup`` is the bridge between a FullFlow network and external code such
    as ThermoProp objects, CoolProp wrappers, interpolation functions, property
    packages, correlations, or user-defined callables. Inputs may be constants,
    :class:`State` objects, derived states, other ``Lookup`` objects, or
    attributes from other lookups.

    The wrapped callable is evaluated during network state evaluation. The
    returned object is stored in ``output`` and its attributes are exposed as
    state-like proxies. For example, after creating ``gas = Lookup(...)``,
    expressions like ``gas.pressure``, ``gas.temperature``, and
    ``gas.enthalpy`` can be passed directly into other components.

    Priority Inputs
    ---------------
    Some property packages allow multiple state-input pairs, but only one value
    from each interchangeable group should be passed at a time. ``priority``
    handles this without requiring full input-mode definitions.

    For each priority group, only the first available value is passed to the
    wrapped callable. A value may come from an existing lookup input or from an
    already evaluated output attribute when the callable accepts that attribute
    name as an input. All inputs not listed in a priority group are passed
    normally.

    Example: initialize an ideal gas with pressure-temperature, then switch to
    pressure-enthalpy once enthalpy becomes a solver variable::

        gas = Lookup(
            "Gas",
            network,
            IdealGas,
            fluid="gn2",
            pressure=3e7,
            temperature=300,
            priority=("enthalpy", "temperature"),
        )

    Initial evaluation passes::

        IdealGas(fluid="gn2", pressure=3e7, temperature=300)

    After ``gas.enthalpy`` becomes assigned, evaluation passes::

        IdealGas(fluid="gn2", pressure=..., enthalpy=...)

    Multiple independent priority groups are supported::

        priority=[
            ("enthalpy", "temperature"),
            ("density", "pressure"),
        ]

    Common Usage
    ------------
    Property object or constructor::

        fuel = Lookup(
            "Fuel",
            network,
            Propellant,
            "rp-1",
            temperature=298.15,
        )

    Chained lookup::

        reactants = Lookup(
            "Reactants",
            network,
            Reactants,
            fuels=fuel,
            oxidizers=oxidizer,
            mixture_ratio=2.0,
        )

        eq = Lookup(
            "Combustion",
            network,
            Equilibrium,
            reactants=reactants,
            pressure=chamber.pressure,
        )

    Parameters
    ----------
    name : str
        Component name shown in reports and diagnostics.
    network : Network
        Network that owns this lookup.
    callable_ : Callable[..., T]
        Function, class, constructor, or callable object to evaluate.
    *args : Any
        Positional arguments passed to ``callable_``. Nested FullFlow states and
        lookup attributes are resolved before evaluation.
    output : State, optional
        State used to store the returned object. Most users can omit this.
    evaluate_on_set : bool, default=False
        If True, changing an input immediately evaluates the lookup.
    strict_inputs : bool, default=False
        If True, assigning an unknown attribute raises instead of creating an
        output guess.
    strict_outputs : bool, default=False
        Reserved for stricter output checking.
    wrap_errors : bool, default=False
        If True, exceptions from the wrapped callable are wrapped with this
        lookup's name.
    evaluate_in_pre_evaluation : bool, default=True
        If True, evaluate during the network pre-evaluation pass.
    lazy : bool, optional
        Convenience inverse of ``evaluate_in_pre_evaluation``.
    defer_until_inputs_available : bool, default=True
        If True, unavailable inputs defer evaluation instead of raising
        immediately.
    cache : bool, default=True
        If True, skip evaluation when resolved inputs and callable structure are
        unchanged.
    cache_tol : float, default=0.0
        Optional numeric tolerance used when building cache fingerprints.
    reuse_existing : bool, default=True
        If True, try to update an existing output object via ``update(...)``
        before constructing a new one.
    memo_size : int, default=1
        Number of recent input/output combinations to retain.
    output_guesses : dict[str, Any], optional
        Initial guesses for output attributes before first evaluation.
    input_guesses : dict[str, Any], optional
        Fallback values for temporarily unavailable inputs.
    priority : tuple[str, ...] or sequence of tuple[str, ...], optional
        Input priority group or groups. For each group, only the first available
        input is passed.
    **kwargs : Any
        Keyword arguments passed to ``callable_``. Keyword arguments also become
        named lookup inputs.

    Notes
    -----
    ``Lookup`` intentionally does not call ``Component.setup()`` because it must
    preserve raw positional and keyword dependency objects. Calling setup would
    wrap constructor arguments in ways that break dynamic lookup inputs.
    """

    _INTERNAL_ATTRS = {
        "name",
        "network",
        "callable",
        "args",
        "kwargs",
        "output",
        "dirty",
        "evaluate_on_set",
        "strict_inputs",
        "strict_outputs",
        "wrap_errors",
        "evaluate_in_pre_evaluation",
        "defer_until_inputs_available",
        "cache",
        "cache_tol",
        "reuse_existing",
        "memo_size",
        "priority",
        "_memo",
        "_signature",
        "_accepted_keywords",
        "_accepts_var_keyword",
        "_output_states",
        "_input_fallback_states",
        "_last_input_key",
        "_last_structure_key",
        "_last_priority",
        "_inactive_priority_inputs",
        "_evaluation_count",
        "_build_count",
        "_reuse_count",
        "_cache_hits",
        "_defer_count",
        "_last_error",
        "_attribute_cache",
    }

    def __init__(
        self,
        name: str,
        network: Network,
        callable_: Callable[..., T],
        *args: Any,
        output: State | None = None,
        evaluate_on_set: bool = False,
        strict_inputs: bool = False,
        strict_outputs: bool = False,
        wrap_errors: bool = False,
        evaluate_in_pre_evaluation: bool = True,
        lazy: bool | None = None,
        defer_until_inputs_available: bool = True,
        cache: bool = True,
        cache_tol: float = 0.0,
        reuse_existing: bool = True,
        memo_size: int = 1,
        output_guesses: dict[str, Any] | None = None,
        input_guesses: dict[str, Any] | None = None,
        priority: (
            tuple[str, ...]
            | list[tuple[str, ...]]
            | tuple[tuple[str, ...], ...]
            | None
        ) = None,
        **kwargs: Any,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "callable", callable_)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "priority", self._normalize_priority(priority))
        object.__setattr__(self, "output", output if output is not None else State())

        if lazy is not None:
            evaluate_in_pre_evaluation = not bool(lazy)

        object.__setattr__(self, "dirty", True)
        object.__setattr__(self, "evaluate_on_set", bool(evaluate_on_set))
        object.__setattr__(self, "strict_inputs", bool(strict_inputs))
        object.__setattr__(self, "strict_outputs", bool(strict_outputs))
        object.__setattr__(self, "wrap_errors", bool(wrap_errors))
        object.__setattr__(self, "evaluate_in_pre_evaluation", bool(evaluate_in_pre_evaluation))
        object.__setattr__(self, "defer_until_inputs_available", bool(defer_until_inputs_available))

        object.__setattr__(self, "cache", bool(cache))
        object.__setattr__(self, "cache_tol", float(cache_tol))
        object.__setattr__(self, "reuse_existing", bool(reuse_existing))
        object.__setattr__(self, "memo_size", max(1, int(memo_size)))
        object.__setattr__(self, "_memo", OrderedDict())

        object.__setattr__(self, "_last_input_key", None)
        object.__setattr__(self, "_last_structure_key", None)
        object.__setattr__(self, "_last_priority", {})
        object.__setattr__(self, "_inactive_priority_inputs", set())
        object.__setattr__(self, "_evaluation_count", 0)
        object.__setattr__(self, "_build_count", 0)
        object.__setattr__(self, "_reuse_count", 0)
        object.__setattr__(self, "_cache_hits", 0)
        object.__setattr__(self, "_defer_count", 0)
        object.__setattr__(self, "_last_error", None)
        object.__setattr__(self, "_attribute_cache", {})

        output_guesses = output_guesses or {}
        input_guesses = input_guesses or {}

        object.__setattr__(
            self,
            "_output_states",
            {
                str(key): value if isinstance(value, State) else State(value)
                for key, value in output_guesses.items()
            },
        )

        object.__setattr__(
            self,
            "_input_fallback_states",
            {
                str(key): value if isinstance(value, State) else State(value)
                for key, value in input_guesses.items()
            },
        )

        self._inspect_callable_signature()

        # Do not call Component.setup().
        # Lookup must preserve raw args/kwargs dependency objects.
        self.initialize_component(name, network)

    @property
    def value(self) -> T:
        return self.output.value

    @property
    def obj(self) -> T:
        return self.output.value

    def __call__(self) -> T:
        return self.output.value

    @staticmethod
    def _value(x: Any) -> Any:
        if isinstance(x, State):
            return x.value

        if isinstance(x, LookupAttribute):
            return x.value

        if isinstance(x, Lookup):
            return x.value

        if isinstance(x, tuple):
            return tuple(Lookup._value(v) for v in x)

        if isinstance(x, list):
            return [Lookup._value(v) for v in x]

        if isinstance(x, dict):
            return {
                Lookup._value(k): Lookup._value(v)
                for k, v in x.items()
            }

        return x

    @staticmethod
    def _is_dynamic_input(x: Any) -> bool:
        if isinstance(x, (State, Lookup, LookupAttribute)):
            return True

        if isinstance(x, tuple):
            return any(Lookup._is_dynamic_input(v) for v in x)

        if isinstance(x, list):
            return any(Lookup._is_dynamic_input(v) for v in x)

        if isinstance(x, dict):
            return any(
                Lookup._is_dynamic_input(k)
                or Lookup._is_dynamic_input(v)
                for k, v in x.items()
            )

        return False

    @staticmethod
    def _normalize_priority(
        priority: (
            tuple[str, ...]
            | list[tuple[str, ...]]
            | tuple[tuple[str, ...], ...]
            | None
        ),
    ) -> tuple[tuple[str, ...], ...]:
        """Normalize priority input selection into tuple-of-tuples form."""
        if priority is None:
            return ()

        if not priority:
            return ()

        # Single group:
        #
        #     priority=("enthalpy", "temperature")
        #
        # becomes:
        #
        #     (("enthalpy", "temperature"),)
        if all(isinstance(item, str) for item in priority):
            return (tuple(str(item) for item in priority),)

        # Multiple independent groups:
        #
        #     priority=[
        #         ("enthalpy", "temperature"),
        #         ("density", "pressure"),
        #     ]
        return tuple(
            tuple(str(name) for name in group)
            for group in priority
        )

    def _priority_candidate_value(self, name: str) -> Any:
        """
        Resolve one candidate from a priority group.

        A priority candidate can come from either:

        1. an existing lookup keyword input, or
        2. an already available output attribute, if the wrapped callable accepts
           that name as an input.

        This second case is what allows property lookups to initialize with an
        easy state pair such as pressure-temperature and later switch to a solver
        state pair such as pressure-enthalpy without the user manually adding an
        ``enthalpy`` keyword input up front.
        """
        if name in self.kwargs:
            return self.get_input(name)

        if self.accepts_input(name):
            return self.get_output(name)

        raise AttributeError(
            f"{self.name!r} has no available priority input or output named {name!r}."
        )

    def _resolve_args_kwargs(self) -> tuple[list[Any], dict[str, Any]]:
        args = [self._value(arg) for arg in self.args]

        if not self.priority:
            kwargs = {
                key: self.get_input(key)
                for key in self.kwargs
            }
            object.__setattr__(self, "_last_priority", {})
            object.__setattr__(self, "_inactive_priority_inputs", set())
            return args, kwargs

        priority_names: set[str] = set()
        selected_inputs: dict[str, Any] = {}
        selected_report: dict[str, str] = {}

        for group_index, group in enumerate(self.priority):
            # Every name in the group is mutually exclusive, so every name in
            # the group must be removed from the normal kwargs pass-through.
            # The previous implementation only added names as it tested them.
            # If the first candidate succeeded, lower-priority names were never
            # excluded, causing e.g. pressure + enthalpy + temperature to be
            # passed to ThermoProp.
            group_names = tuple(group)
            priority_names.update(group_names)

            chosen_name = None
            chosen_value = None

            for name in group_names:
                try:
                    chosen_value = self._priority_candidate_value(name)
                except Exception:
                    continue

                chosen_name = name
                break

            if chosen_name is not None:
                selected_inputs[chosen_name] = chosen_value
                selected_report[f"group_{group_index}"] = chosen_name

        kwargs = {
            key: self.get_input(key)
            for key in self.kwargs
            if key not in priority_names
        }
        kwargs.update(selected_inputs)

        inactive_priority_inputs = {
            name
            for name in priority_names
            if name in self.kwargs and name not in selected_inputs
        }

        object.__setattr__(self, "_last_priority", selected_report)
        object.__setattr__(self, "_inactive_priority_inputs", inactive_priority_inputs)

        return args, kwargs

    def _inspect_callable_signature(self) -> None:
        cache_key = id(self.callable)
        cached = _SIGNATURE_CACHE.get(cache_key)

        if cached is None:
            accepted_keywords: set[str] = set()
            accepts_var_keyword = False
            signature = None

            try:
                signature = inspect.signature(self.callable)
            except (TypeError, ValueError):
                pass

            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.kind == parameter.VAR_KEYWORD:
                        accepts_var_keyword = True
                    elif parameter.kind in {
                        parameter.POSITIONAL_OR_KEYWORD,
                        parameter.KEYWORD_ONLY,
                    }:
                        accepted_keywords.add(parameter.name)

            cached = (signature, accepted_keywords, accepts_var_keyword)
            _SIGNATURE_CACHE[cache_key] = cached

        signature, accepted_keywords, accepts_var_keyword = cached
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(self, "_accepted_keywords", accepted_keywords)
        object.__setattr__(self, "_accepts_var_keyword", accepts_var_keyword)

    def accepts_input(self, name: str) -> bool:
        if name in self.kwargs:
            return True

        return self._accepts_var_keyword or name in self._accepted_keywords

    def has_input(self, name: str) -> bool:
        return name in self.kwargs

    def input_is_active(self, name: str) -> bool:
        """Return True when a raw keyword input is currently active.

        Priority groups can temporarily deactivate lower-priority keyword
        inputs. For example, with ``priority=("enthalpy", "temperature")``,
        ``temperature`` is a construction-time input during the initial
        pressure-temperature evaluation, but becomes an output once enthalpy is
        selected.
        """
        return name in self.kwargs and name not in self._inactive_priority_inputs

    def input_is_assigned(self, name: str) -> bool:
        if name not in self.kwargs:
            return False

        try:
            self.get_input(name)
            return True
        except Exception:
            return False

    def get_input(self, name: str) -> Any:
        if name not in self.kwargs:
            raise AttributeError(
                f"{self.name!r} has no callable input named {name!r}."
            )

        try:
            value = self._value(self.kwargs[name])
        except Exception as exc:
            fallback = self._input_fallback_states.get(name)

            if fallback is not None and fallback.is_assigned:
                return fallback.value

            raise ValueError(
                f"{self.name!r}.{name} is not available yet and has no "
                "input fallback / initial guess."
            ) from exc

        fallback = self._input_fallback_states.get(name)

        if fallback is not None:
            try:
                fallback.value = value
            except Exception:
                pass

        return value

    def set_input(self, name: str, value: Any) -> None:
        if not self.accepts_input(name):
            raise AttributeError(
                f"{self.name!r} cannot accept {name!r} as an input for "
                f"{getattr(self.callable, '__name__', repr(self.callable))}."
            )

        current = self.kwargs.get(name, _MISSING)

        old_value_available = False
        old_value = None

        if current is not _MISSING:
            try:
                old_value = self.get_input(name)
                old_value_available = True
            except Exception:
                pass

        old_storage_key = self._storage_key(current)
        old_value_key = self._safe_resolved_key(current)

        new_is_dynamic = self._is_dynamic_input(value)

        if new_is_dynamic and old_value_available:
            fallback = self._input_fallback_states.get(name)

            if fallback is None:
                self._input_fallback_states[name] = State(old_value)
            else:
                try:
                    fallback.value = old_value
                except Exception:
                    pass

        elif not new_is_dynamic:
            fallback = self._input_fallback_states.get(name)

            if fallback is not None:
                try:
                    fallback.value = value
                except Exception:
                    pass

        if (
            current is not _MISSING
            and isinstance(current, State)
            and not new_is_dynamic
        ):
            current.value = value
        elif (
            current is _MISSING
            and not new_is_dynamic
            and name in self._output_states
        ):
            # Promote an output fallback/guess State to a real callable input.
            #
            # This preserves bounds and keep_feasible for lookup outputs that
            # later become solver variables. Example: initialize a property
            # lookup with pressure-temperature, then let the solver drive
            # pressure-enthalpy.
            state = self._output_states[name]
            state.value = value
            self.kwargs[name] = state
        else:
            self.kwargs[name] = value

        new_storage_key = self._storage_key(self.kwargs.get(name, _MISSING))
        new_value_key = self._safe_resolved_key(self.kwargs.get(name, _MISSING))

        changed = (
            old_storage_key != new_storage_key
            or old_value_key != new_value_key
        )

        if changed:
            self.dirty = True

        if changed and self.evaluate_on_set:
            self.evaluate_states()

    def update(self, **kwargs: Any) -> "Lookup[T]":
        for name, value in kwargs.items():
            self.set_input(name, value)

        return self

    def input(self, name: str) -> LookupAttribute:
        if not self.has_input(name) and not self.accepts_input(name):
            raise AttributeError(f"{self.name!r} has no input named {name!r}.")

        attribute = self._attribute_cache.get(name)
        if attribute is None:
            attribute = LookupAttribute(self, name)
            self._attribute_cache[name] = attribute
        return attribute

    def input_state(self, name: str, default: Any = None) -> State:
        if self.has_input(name):
            current = self.kwargs[name]

            if isinstance(current, State):
                return current

            try:
                state = State(self.get_input(name))
            except Exception:
                state = State(default)

            self.kwargs[name] = state
            self.dirty = True
            return state

        if not self.accepts_input(name):
            raise AttributeError(f"{self.name!r} cannot accept {name!r} as an input.")

        state = State(default)
        self.kwargs[name] = state
        self.dirty = True
        return state

    def input_fallback_state(self, name: str, default: Any = _MISSING) -> State:
        if name not in self._input_fallback_states:
            if default is _MISSING:
                self._input_fallback_states[name] = State()
            else:
                self._input_fallback_states[name] = State(default)

        return self._input_fallback_states[name]

    def output_state(self, name: str, default: Any = _MISSING) -> State:
        if name not in self._output_states:
            if default is _MISSING:
                self._output_states[name] = State()
            else:
                self._output_states[name] = State(default)

        return self._output_states[name]

    def set_output_guess(self, name: str, value: Any) -> None:
        if isinstance(value, State):
            self._output_states[name] = value
            return

        self.output_state(name).value = value

    def output_is_assigned(self, name: str) -> bool:
        try:
            obj = self.output.value
            getattr(obj, name)
            return True
        except Exception:
            pass

        state = self._output_states.get(name)
        return bool(state is not None and state.is_assigned)

    def get_output(self, name: str) -> Any:
        try:
            obj = self.output.value
            value = getattr(obj, name)
        except Exception as exc:
            state = self._output_states.get(name)

            if state is not None and state.is_assigned:
                return state.value

            raise ValueError(
                f"{self.name!r}.{name} is not available yet. "
                "Evaluate the lookup first, place it earlier in the network, "
                "or provide an output guess with "
                f"{self.name}.output_state({name!r}, guess) or "
                f"{self.name}.{name}.value = guess."
            ) from exc

        if name in self._output_states:
            try:
                self._output_states[name].value = value
            except Exception:
                pass

        return value

    def state(self, name: str, default: Any = _MISSING) -> State:
        if self.has_input(name) or self.accepts_input(name):
            return State._derived(lambda: self.get_input(name))

        if default is not _MISSING:
            self.output_state(name, default)

        return State._derived(lambda: self.get_output(name))

    def set_arg(self, index: int, value: Any) -> None:
        old_key = self._safe_resolved_key(self.args)

        args = list(self.args)
        args[index] = value
        self.args = tuple(args)

        new_key = self._safe_resolved_key(self.args)

        if old_key != new_key:
            self.dirty = True

        if self.dirty and self.evaluate_on_set:
            self.evaluate_states()

    def _input_key(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        return (
            tuple(self._fingerprint_value(arg) for arg in args),
            tuple(
                (key, self._fingerprint_value(value))
                for key, value in sorted(kwargs.items())
            ),
        )

    def _structure_key(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        return (
            getattr(self.callable, "__module__", None),
            getattr(self.callable, "__qualname__", repr(self.callable)),
            tuple(self._structure_fingerprint(arg, positional=True) for arg in args),
            tuple(
                (key, self._structure_fingerprint(value, positional=False))
                for key, value in sorted(kwargs.items())
                if self._is_identity_like(value)
            ),
        )

    def _safe_resolved_key(self, value: Any) -> Any:
        if value is _MISSING:
            return _MISSING

        try:
            resolved = self._value(value)
        except Exception:
            return _UNAVAILABLE

        return self._fingerprint_value(resolved)

    def _storage_key(self, value: Any) -> Any:
        if value is _MISSING:
            return _MISSING

        if isinstance(value, State):
            return ("State", id(value))

        if isinstance(value, Lookup):
            return ("Lookup", id(value))

        if isinstance(value, LookupAttribute):
            return ("LookupAttribute", id(value.lookup), value.name)

        if isinstance(value, tuple):
            return ("tuple", tuple(self._storage_key(v) for v in value))

        if isinstance(value, list):
            return ("list", tuple(self._storage_key(v) for v in value))

        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (self._storage_key(k), self._storage_key(v))
                    for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
                ),
            )

        return ("plain", type(value).__name__)

    def _fingerprint_value(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, bytes, bool)):
            return value

        if isinstance(value, (int, float, np.number)):
            v = float(value)

            if math.isnan(v):
                return ("nan",)

            if math.isinf(v):
                return ("inf", 1 if v > 0.0 else -1)

            if self.cache_tol > 0.0:
                return round(v / self.cache_tol)

            return v

        if isinstance(value, tuple):
            return tuple(self._fingerprint_value(v) for v in value)

        if isinstance(value, list):
            return tuple(self._fingerprint_value(v) for v in value)

        if isinstance(value, dict):
            return tuple(
                (self._fingerprint_value(k), self._fingerprint_value(v))
                for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
            )

        if isinstance(value, np.ndarray):
            return (
                "ndarray",
                value.shape,
                str(value.dtype),
                value.tobytes(),
            )

        if hasattr(value, "cache_key") and callable(value.cache_key):
            try:
                return ("cache_key", value.cache_key())
            except Exception:
                pass

        try:
            hash(value)
        except Exception:
            return ("object", type(value).__name__, id(value))

        if hasattr(value, "__dict__"):
            return ("object", type(value).__name__, id(value))

        return ("hashable", type(value).__name__, value)

    def _is_identity_like(self, value: Any) -> bool:
        if isinstance(value, (int, float, np.number)):
            return False

        return True

    def _structure_fingerprint(self, value: Any, positional: bool) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, bytes, bool)):
            return value

        if isinstance(value, (int, float, np.number)):
            if positional:
                return self._fingerprint_value(value)
            return "<numeric>"

        if isinstance(value, tuple):
            return tuple(
                self._structure_fingerprint(v, positional=positional)
                for v in value
            )

        if isinstance(value, list):
            return tuple(
                self._structure_fingerprint(v, positional=positional)
                for v in value
            )

        if isinstance(value, dict):
            return tuple(
                (
                    self._structure_fingerprint(k, positional=True),
                    self._structure_fingerprint(v, positional=positional),
                )
                for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
            )

        return ("object", type(value).__name__, id(value))

    def clear_cache(self) -> None:
        self._last_input_key = None
        self._last_structure_key = None
        self._memo.clear()
        self.dirty = True

    def mark_dirty(self) -> None:
        self.dirty = True

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def build_count(self) -> int:
        return self._build_count

    @property
    def reuse_count(self) -> int:
        return self._reuse_count

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def defer_count(self) -> int:
        return self._defer_count

    def cache_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "callable": getattr(self.callable, "__name__", repr(self.callable)),
            "cache": self.cache,
            "reuse_existing": self.reuse_existing,
            "memo_size": self.memo_size,
            "priority": self.priority,
            "last_priority": self._last_priority,
            "inactive_priority_inputs": tuple(sorted(self._inactive_priority_inputs)),
            "memo_entries": len(self._memo),
            "dirty": self.dirty,
            "evaluation_count": self._evaluation_count,
            "build_count": self._build_count,
            "reuse_count": self._reuse_count,
            "cache_hits": self._cache_hits,
            "defer_count": self._defer_count,
            "last_error": None if self._last_error is None else repr(self._last_error),
        }

    def _memo_key(self, structure_key: Any, input_key: Any) -> Any:
        return structure_key, input_key

    def _get_memoized(self, structure_key: Any, input_key: Any) -> Any:
        key = self._memo_key(structure_key, input_key)
        try:
            result = self._memo.pop(key)
        except KeyError:
            return _MISSING
        self._memo[key] = result
        return result

    def _remember_result(self, structure_key: Any, input_key: Any, result: Any) -> None:
        if not self.cache:
            return
        key = self._memo_key(structure_key, input_key)
        self._memo[key] = result
        while len(self._memo) > self.memo_size:
            self._memo.popitem(last=False)

    def pre_evaluation(self) -> None:
        if self.evaluate_in_pre_evaluation:
            self.evaluate_states()

    def evaluate_states(self) -> None:
        try:
            args, kwargs = self._resolve_args_kwargs()
        except Exception as exc:
            self._last_error = exc

            if self.defer_until_inputs_available:
                self._defer_count += 1
                self.dirty = True
                return

            raise

        input_key = self._input_key(args, kwargs)
        structure_key = self._structure_key(args, kwargs)

        if (
            self.cache
            and not self.dirty
            and self.output.is_assigned
            and input_key == self._last_input_key
            and structure_key == self._last_structure_key
        ):
            self._cache_hits += 1
            self._refresh_output_states()
            return

        if self.cache:
            memoized_result = self._get_memoized(structure_key, input_key)
            if memoized_result is not _MISSING:
                self.output.value = memoized_result
                self._last_input_key = input_key
                self._last_structure_key = structure_key
                self._cache_hits += 1
                self._last_error = None
                self._refresh_output_states()
                self.dirty = False
                return

        can_reuse = (
            self.reuse_existing
            and self.output.is_assigned
            and structure_key == self._last_structure_key
        )

        if can_reuse:
            try:
                result = self._update_existing_output(args, kwargs)
                self._reuse_count += 1
            except NotImplementedError:
                result = self._call_new(args, kwargs)
                self._build_count += 1
            except Exception:
                result = self._call_new(args, kwargs)
                self._build_count += 1
        else:
            result = self._call_new(args, kwargs)
            self._build_count += 1

        self.output.value = result
        self._remember_result(structure_key, input_key, result)
        self._last_input_key = input_key
        self._last_structure_key = structure_key
        self._evaluation_count += 1
        self._last_error = None
        self._refresh_output_states()
        self.dirty = False

    def _call_new(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if self.wrap_errors:
            try:
                return self.callable(*args, **kwargs)
            except Exception as exc:
                self._last_error = exc
                callable_name = getattr(self.callable, "__name__", repr(self.callable))
                raise RuntimeError(
                    f"{self.name}: failed to evaluate {callable_name}."
                ) from exc

        return self.callable(*args, **kwargs)

    def _update_existing_output(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        obj = self.output.value

        update = getattr(obj, "update", None)

        if update is None or not callable(update):
            raise NotImplementedError

        update_kwargs = dict(kwargs)

        update_type = type(obj)
        accepts_solve = _UPDATE_ACCEPTS_SOLVE_CACHE.get(update_type)
        if accepts_solve is None:
            try:
                accepts_solve = "solve" in inspect.signature(update).parameters
            except (TypeError, ValueError):
                accepts_solve = False
            _UPDATE_ACCEPTS_SOLVE_CACHE[update_type] = accepts_solve

        if accepts_solve and "solve" not in update_kwargs:
            update_kwargs["solve"] = False

        update_result = update(**update_kwargs)

        if update_result is not None:
            obj = update_result

        solve = getattr(obj, "solve", None)

        if callable(solve):
            try:
                solve_result = solve()

                if solve_result is not None:
                    obj = solve_result

            except TypeError:
                pass

        return obj

    def _refresh_output_states(self) -> None:
        if not self._output_states:
            return

        try:
            obj = self.output.value
        except Exception:
            return

        for name, state in self._output_states.items():
            try:
                state.value = getattr(obj, name)
            except Exception:
                pass

    def __getattr__(self, name: str) -> LookupAttribute:
        if name.startswith("__"):
            raise AttributeError(name)

        attribute = self._attribute_cache.get(name)
        if attribute is None:
            attribute = LookupAttribute(self, name)
            self._attribute_cache[name] = attribute
        return attribute

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in self._INTERNAL_ATTRS
            or name.startswith("_")
            or "kwargs" not in self.__dict__
        ):
            object.__setattr__(self, name, value)
            return

        if name in self.kwargs or self.accepts_input(name):
            self.set_input(name, value)
            return

        if self.strict_inputs:
            raise AttributeError(
                f"{self.name!r} has no input named {name!r}. "
                "Use strict_inputs=False or assign an output guess with "
                f"{self.name}.output_state({name!r}, guess)."
            )

        self.set_output_guess(name, value)

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        names.update(self.kwargs.keys())
        names.update(self._output_states.keys())
        names.update(self._input_fallback_states.keys())

        try:
            names.update(dir(self.output.value))
        except Exception:
            pass

        try:
            names.update(dir(self.callable))
        except Exception:
            pass

        return sorted(names)

    @property
    def wrapped_signature(self):
        return self._signature

    @property
    def wrapped_doc(self) -> str | None:
        return inspect.getdoc(self.callable)

    def help(self) -> None:
        callable_name = getattr(self.callable, "__name__", repr(self.callable))
        print(f"{self.name}: {callable_name}")

        if self._signature is not None:
            print(self._signature)

        doc = inspect.getdoc(self.callable)
        if doc:
            print()
            print(doc)

    @property
    def ignored_export_attributes(self) -> set[str]:
        return {
            "callable",
            "args",
            "kwargs",
            "output",
            "dirty",
            "evaluate_on_set",
            "strict_inputs",
            "strict_outputs",
            "wrap_errors",
            "evaluate_in_pre_evaluation",
            "defer_until_inputs_available",
            "cache",
            "cache_tol",
            "reuse_existing",
            "memo_size",
            "priority",
            "_memo",
            "_signature",
            "_accepted_keywords",
            "_accepts_var_keyword",
            "_output_states",
            "_input_fallback_states",
            "_last_input_key",
            "_last_structure_key",
            "_last_priority",
            "_inactive_priority_inputs",
            "_evaluation_count",
            "_build_count",
            "_reuse_count",
            "_cache_hits",
            "_defer_count",
            "_last_error",
            "_attribute_cache",
        }

    def __repr__(self) -> str:
        try:
            value_repr = repr(self.output.value)
        except Exception:
            value_repr = "<unevaluated>"

        callable_name = getattr(self.callable, "__name__", repr(self.callable))

        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"callable={callable_name}, "
            f"dirty={self.dirty}, "
            f"evals={self._evaluation_count}, "
            f"builds={self._build_count}, "
            f"reuses={self._reuse_count}, "
            f"cache_hits={self._cache_hits}, "
            f"value={value_repr})"
        )

# Backward-compatible aliases.
#
# ``Lookup`` and ``LookupAttribute`` are the canonical names. The old names are
# kept so existing FullFlow scripts that use ``CallableLookup`` continue to run.
CallableLookup = Lookup
CallableLookupAttribute = LookupAttribute

__all__ = [
    "Lookup",
    "LookupAttribute",
    "CallableLookup",
    "CallableLookupAttribute",
]
