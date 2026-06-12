from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class CallableLookup(Component):
    """
    Evaluate a callable every network state evaluation.

    CallableLookup provides a generic interface for rebuilding objects or
    evaluating functions whose inputs depend on FullFlow States. Any State
    objects passed through positional or keyword arguments are automatically
    replaced with their current values before the callable is executed.

    This component can be used to construct ThermoProp objects, perform
    calculations, generate lookup tables, or wrap arbitrary user-defined
    functions.

    Examples
    --------

    Function evaluation:

    >>> area = CallableLookup(
    ...     "Area",
    ...     network,
    ...     lambda d: np.pi * d**2 / 4,
    ...     diameter,
    ... )

    Object construction:

    >>> fuel = CallableLookup(
    ...     "Fuel",
    ...     network,
    ...     Propellant,
    ...     "RP-1",
    ...     temperature=fuel_temperature,
    ...     pressure=fuel_pressure,
    ... )

    Chained object construction:

    >>> reactants = CallableLookup(
    ...     "Reactants",
    ...     network,
    ...     Reactants,
    ...     fuels=fuel.output,
    ...     oxidizers=ox.output,
    ...     mixture_ratio=MR,
    ... )

    Parameters
    ----------
    callable_ : callable
        Function or class constructor to evaluate.

    *args
        Positional arguments passed to the callable.

    **kwargs
        Keyword arguments passed to the callable.

    output : State, optional
        State used to store the callable result. If omitted, an internal
        State is created automatically.

    Outputs
    -------
    output : State
        State containing the callable result.

    value : Any
        Convenience alias for ``output.value``.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        callable_: Callable[..., Any],
        *args: Any,
        output: State | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self.network = network

        self.callable = callable_
        self.args = args
        self.kwargs = kwargs

        self.output = output if output is not None else State(None)

        self.setup()

    @property
    def value(self) -> Any:
        """
        Convenience alias for output.value.
        """
        return self.output.value

    @staticmethod
    def _value(x: Any) -> Any:
        """
        Recursively replace States with their current values.
        """

        if isinstance(x, State):
            return x.value

        if isinstance(x, tuple):
            return tuple(CallableLookup._value(v) for v in x)

        if isinstance(x, list):
            return [CallableLookup._value(v) for v in x]

        if isinstance(x, dict):
            return {
                CallableLookup._value(k): CallableLookup._value(v)
                for k, v in x.items()
            }

        return x

    def evaluate_states(self):
        """
        Evaluate the callable using current State values.
        """

        args = [self._value(arg) for arg in self.args]

        kwargs = {
            key: self._value(value)
            for key, value in self.kwargs.items()
        }

        self.output.value = self.callable(*args, **kwargs)

    def __getattr__(self, name: str):
        """
        Forward unknown attribute access to the evaluated object.

        Examples
        --------

        >>> gas.enthalpy
        >>> gas.density
        >>> fuel.temperature

        instead of

        >>> gas.output.value.enthalpy
        >>> gas.output.value.density
        >>> fuel.output.value.temperature
        """

        try:
            obj = self.output.value
        except Exception as exc:
            raise AttributeError(
                f"{self.__class__.__name__!s} has no evaluated value yet."
            ) from exc

        return getattr(obj, name)

    def __call__(self) -> Any:
        """
        Return the current evaluated value.

        Examples
        --------

        >>> fuel()
        >>> chamber()
        >>> gas()
        """

        return self.output.value

    def __repr__(self) -> str:
        try:
            value_repr = repr(self.output.value)
        except Exception:
            value_repr = "<unevaluated>"

        return (
            f"{self.__class__.__name__}("
            f"callable={getattr(self.callable, '__name__', repr(self.callable))}, "
            f"value={value_repr})"
        )