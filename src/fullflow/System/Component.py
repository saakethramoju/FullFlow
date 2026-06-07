from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import inspect

from .State import State
from .Composition import Composition
from .Model import ModelOption

if TYPE_CHECKING:
    from fullflow.System import Network

class Component(ABC):

    def __init__(self, 
                 name: str,
                 network: Network):
        
        self.setup()

    def setup(self) -> None:
        """
        Automatically create component attributes from constructor arguments.

        Default Composition() arguments are replaced with fresh instances so
        components do not accidentally share the same Composition object.
        """

        frame = inspect.currentframe().f_back

        if frame is None:
            raise RuntimeError("Could not access caller frame.")

        local_vars = frame.f_locals

        # Required constructor arguments.
        name = local_vars["name"]
        network = local_vars["network"]

        self.initialize_component(name, network)

        signature = inspect.signature(self.__class__.__init__)

        for attr_name, value in local_vars.items():

            if attr_name in {"self", "name", "network"}:
                continue

            parameter = signature.parameters.get(attr_name)

            # True only when the constructor argument was not overridden.
            is_default_value = (
                parameter is not None
                and parameter.default is not inspect.Parameter.empty
                and value is parameter.default
            )

            setattr(
                self,
                attr_name,
                self.initialize_attribute(
                    value,
                    is_default_value,
                ),
            )


    def initialize_attribute(
        self,
        value: State | Composition | float | int | str | bool | None = None,
        is_default_value: bool = False,
    ):
        """
        Normalize constructor inputs.

        State        -> keep as-is
        Composition  -> keep as-is
        float/int    -> convert to State
        None         -> empty State
        bool/str     -> keep as-is

        Default Composition() arguments are replaced with fresh empty
        Composition instances.
        """

        # Prevent shared default Composition() objects.
        if is_default_value and isinstance(value, Composition):
            return Composition()

        if isinstance(value, State):
            return value

        if isinstance(value, Composition):
            return value

        if value is None:
            return State()

        if isinstance(value, bool):
            return value

        if isinstance(value, (float, int)):
            return State(float(value))

        return value



    def initialize_component(self, name: str, network: Network) -> None:
        self.name = name
        self.network = network
        self.network.add_component(component=self)



    @classmethod
    def model(cls, name: str | None = None, **kwargs):
        """
        Create a deferred model option for this component class.

        This does not construct the component or register it with a Network.
        The component is built later by Model.
        """

        return ModelOption(
            name or cls.__name__,
            component_class=cls,
            kwargs=kwargs,
        )


    #@abstractmethod
    def pre_evaluation(self) -> None:
        pass

    # never put a derived State in the solver’s iteration-variable list!!!
    @property
    #@abstractmethod
    def iteration_variables(self) -> list[State]:
        return []

    #@abstractmethod
    def evaluate_states(self) -> None:
        pass

    @property
    #@abstractmethod
    def residuals(self) -> list[float]:
        return []



    def __str__(self):

        def format_value(value):
            if isinstance(value, State):
                if not value.is_assigned:
                    return "<uninitialized>"

                try:
                    return value.value
                except Exception:
                    return "<unavailable>"

            elif isinstance(value, list):
                return [format_value(item) for item in value]

            elif isinstance(value, tuple):
                return tuple(format_value(item) for item in value)

            elif isinstance(value, dict):
                return {
                    k: format_value(v)
                    for k, v in value.items()
                }

            else:
                return value

        lines = [f"Component {self.name} ({self.__class__.__name__})"]

        skip_attrs = {"network"} | self.ignored_export_attributes

        for attr, value in self.__dict__.items():

            if attr in skip_attrs:
                continue

            formatted_value = format_value(value)

            lines.append(f"    {attr}: {formatted_value}")

        return "\n".join(lines)
        
    @property
    def ignored_export_attributes(self) -> set[str]:
        return set()
    
    # ---- transient stuff ----#
    @property
    def timestep_variables(self) -> list[State]:
        return []
    
    @property
    def time_derivative(self) -> list[State]:
        return []