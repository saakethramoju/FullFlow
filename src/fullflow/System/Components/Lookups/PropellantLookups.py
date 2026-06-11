from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State
from thermoprop import Propellant

from fullflow.Exceptions import InvalidThermoStateError

if TYPE_CHECKING:
    from fullflow.System import Network


PropellantInput = str


class PropellantLookup(Component):
    """
    RocketProps-backed liquid propellant property lookup component.
    """
    _THERMO_NAMES = (
        "pressure",
        "temperature",
    )

    def __init__(
        self,
        name: str,
        network: Network,
        propellant: PropellantInput,
        temperature: State | float | None = None,
        pressure: State | float | None = None,
        **property_states: State,
    ):

        _input_map = {
            "pressure": pressure,
            "temperature": temperature,
        }

        self.setup()

        if hasattr(self, "_input_map"):
            delattr(self, "_input_map")

        if hasattr(self, "property_states"):
            delattr(self, "property_states")

        if not isinstance(propellant, str):
            raise TypeError(
                f"{self.name}: PropellantLookup only accepts a propellant "
                "name or alias string. Composition objects and multi-species "
                "mixtures are not supported. Use named mixtures such as "
                "'MON25', 'A50', or 'MHF3'."
            )

        self._propellant_input = propellant
        self._propellant_name = propellant

        # Public identity. This is a string, not a Composition.
        self.propellant = self._propellant_name

        self._Propellant = None

        provided_names = [
            prop_name
            for prop_name in self._THERMO_NAMES
            if _input_map[prop_name] is not None
        ]

        if "temperature" not in provided_names:
            raise ValueError(
                "PropellantLookup requires temperature so the initial "
                "liquid propellant state can be defined."
            )

        if "pressure" in provided_names:
            self._flash_names = ["pressure", "temperature"]
        else:
            self._flash_names = ["temperature"]

        self._last_flash_values: tuple[float, ...] | None = None
        self._property_cache: dict[str, float] = {}

        self._property_states: dict[str, State] = {}
        self._external_property_names: set[str] = set()

        self._initialize_backend()

        for flash_name in self._flash_names:
            state = getattr(self, flash_name, None)

            if hasattr(state, "is_assigned"):
                if self._Propellant is not None and not state.is_assigned:
                    state.value = self._get_cached_property(flash_name)
            else:
                setattr(self, flash_name, State(state))

        for prop_name in self._THERMO_NAMES:
            if prop_name in self._flash_names:
                continue

            if _input_map[prop_name] is None and prop_name in self.__dict__:
                delattr(self, prop_name)

        for prop_name in self._THERMO_NAMES:
            if prop_name in self._flash_names:
                continue

            if prop_name in self.__dict__:
                self._property_states[prop_name] = getattr(self, prop_name)
                self._external_property_names.add(prop_name)

        for prop_name, state in property_states.items():

            state = self.initialize_attribute(state)

            if not isinstance(state, State):
                raise TypeError(
                    f"{prop_name!r} must be a State, "
                    f"got {type(state).__name__}."
                )

            if prop_name in self._flash_names:
                raise ValueError(
                    f"{prop_name!r} is already being used as a propellant "
                    f"input and cannot also be used as an output property State."
                )

            if not self._is_propellant_property(prop_name):
                raise AttributeError(
                    f"{prop_name!r} is not a valid Propellant property."
                )

            self._property_states[prop_name] = state
            self._external_property_names.add(prop_name)

    @property
    def propellant_name(self) -> str:
        """Return the canonical ThermoProp propellant name."""
        return self._propellant_name

    @property
    def composition(self):
        raise AttributeError(
            f"{self.name}: PropellantLookup does not support composition. "
            "Pass propellants as strings, or use .propellant_name to pass the "
            "same propellant into another PropellantLookup."
        )

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self) -> None:
        try:
            self._set_propellant_from_flash()

        except Exception as e:
            flash_state = {
                name: getattr(self, name).value
                for name in self._flash_names
            }

            raise InvalidThermoStateError(
                f"{self.name}: invalid propellant state.\n"
                f"Propellant: {self._propellant_name!r}\n"
                f"Flash variables: {flash_state}"
            ) from e

        for prop_name in self._external_property_names:
            self._property_states[prop_name].value = self._get_cached_property(
                prop_name
            )

    def __getattr__(self, name: str) -> State:

        if name == "composition":
            raise AttributeError(
                f"{self.name}: PropellantLookup does not support composition. "
                "Pass propellants as strings, or use .propellant_name to pass the "
                "same propellant into another PropellantLookup."
            )

        if "_Propellant" not in self.__dict__:
            raise AttributeError(name)

        if not self._is_propellant_property(name):
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        if name not in self._property_states:
            self._property_states[name] = State._derived(
                lambda prop=name: self._get_cached_property(prop)
            )

        return self._property_states[name]

    def _initialize_backend(self) -> None:
        """Create the ThermoProp Propellant object."""
        self._Propellant = Propellant(
            self._propellant_input,
            **{
                name: getattr(self, name).value
                for name in self._flash_names
            },
        )

        self._propellant_name = self._Propellant.propellant
        self.propellant = self._propellant_name

        self._last_flash_values = None
        self._property_cache.clear()

    def _set_propellant_from_flash(self) -> None:

        flash_values = tuple(
            getattr(self, prop_name).value
            for prop_name in self._flash_names
        )

        if self._flash_values_unchanged(flash_values):
            return

        if self._flash_names == ["temperature"]:
            self._Propellant.temperature = flash_values[0]

        else:
            self._Propellant.pressure_temperature = flash_values

        self._last_flash_values = flash_values
        self._property_cache.clear()

    def _flash_values_unchanged(
        self,
        flash_values: tuple[float, ...],
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> bool:

        if self._last_flash_values is None:
            return False

        return all(
            np.isclose(current, previous, rtol=rtol, atol=atol)
            for current, previous in zip(
                flash_values,
                self._last_flash_values,
            )
        )

    def _get_cached_property(self, name: str):

        if self._Propellant is None:
            raise ValueError(
                f"{self.name}: cannot evaluate {name!r} because the "
                "propellant backend is not initialized."
            )

        if name not in self._property_cache:
            self._property_cache[name] = getattr(self._Propellant, name)

        return self._property_cache[name]

    def _is_propellant_property(self, name: str) -> bool:
        return isinstance(
            getattr(Propellant, name, None),
            property,
        )

    @classmethod
    def supported_properties(cls) -> list[str]:
        return Propellant.supported_properties()

    @classmethod
    def show_supported_properties(cls) -> list[str]:
        return Propellant.show_supported_properties()

    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        return Propellant.supports_property(property_name)

    @classmethod
    def supported_inputs(cls) -> list[str]:
        return list(cls._THERMO_NAMES)

    @classmethod
    def show_supported_inputs(cls) -> list[str]:
        inputs = cls.supported_inputs()

        for name in inputs:
            print(name)

        return inputs

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        return [
            "temperature",
            "pressure-temperature",
        ]

    @classmethod
    def show_supported_flash_pairs(cls) -> list[str]:
        pairs = cls.supported_flash_pairs()

        for pair in pairs:
            print(pair)

        return pairs

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {
            "property_states",
            "_property_states",
            "external_property_names",
            "_external_property_names",
            "flash_names",
            "_flash_names",
            "Propellant",
            "_Propellant",
            "input_map",
            "_input_map",
            "last_flash_values",
            "_last_flash_values",
            "property_cache",
            "_property_cache",
            "propellant_input",
            "_propellant_input",
            "propellant_name",
            "_propellant_name",
        }