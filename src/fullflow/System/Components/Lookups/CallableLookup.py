from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class CallableLookup(Component):
    """
    Rebuild an object or evaluate a function every state evaluation.
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

    @staticmethod
    def _value(x):
        if hasattr(x, "value"):
            return x.value

        if isinstance(x, tuple):
            return tuple(CallableLookup._value(item) for item in x)

        if isinstance(x, list):
            return [CallableLookup._value(item) for item in x]

        if isinstance(x, dict):
            return {
                CallableLookup._value(key): CallableLookup._value(value)
                for key, value in x.items()
            }

        return x

    def evaluate_states(self):
        args = [self._value(arg) for arg in self.args]
        kwargs = {key: self._value(value) for key, value in self.kwargs.items()}

        self.output.value = self.callable(*args, **kwargs)