from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State
from thermoprop import Material

from fullflow.Exceptions import InvalidMaterialPropertyError

if TYPE_CHECKING:
    from fullflow.System import Network


MaterialInput = str

class MaterialLookup(Component):
    """
    ThermoProp Material-backed solid material property lookup component.

    `MaterialLookup` evaluates temperature-dependent solid material properties
    through the ThermoProp `Material` wrapper. The component updates the
    material backend from a temperature input and writes requested material
    property outputs to their corresponding states.

    Additional supported material properties can be accessed lazily as derived
    states.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    material : str
        Material name or alias
    temperature : State or float, optional
        Material temperature input
    allow_extrapolation : bool, optional
        Whether material property extrapolation is allowed
    **property_states : State
        Additional requested Material property output states

    Outputs
    -------
    property_states : State
        Requested material property states

    Notes
    -----
    Material properties are temperature-dependent only.

    If temperature is omitted, a default temperature input state is created:

        ``temperature = 293.15``

    Requested material properties may be passed with keyword arguments:

        ``MaterialLookup(..., material="c101", temperature=T, thermal_conductivity=k)``

    The `material_name` property returns the canonical ThermoProp material name:

        ``material_name = lookup.material_name``

    Supported properties can be inspected with:

        ``MaterialLookup.supported_properties()``

        ``MaterialLookup.show_supported_properties()``

        ``MaterialLookup.supports_property(property_name)``

    Supported inputs can be inspected with:

        ``MaterialLookup.supported_inputs()``

        ``MaterialLookup.show_supported_inputs()``

    Supported input pairs can be inspected with:

        ``MaterialLookup.supported_flash_pairs()``

        ``MaterialLookup.show_supported_flash_pairs()``

    If the material state is invalid, `InvalidMaterialPropertyError` is raised
    with the material name and temperature value.
    """
    _THERMO_NAMES = (
        "temperature",
    )

    def __init__(
        self,
        name: str,
        network: Network,
        material: MaterialInput,
        temperature: State | float | None = None,
        allow_extrapolation: bool = True,
        **property_states: State,
    ):

        _input_map = {
            "temperature": temperature,
        }

        self.setup()

        if hasattr(self, "_input_map"):
            delattr(self, "_input_map")

        if hasattr(self, "property_states"):
            delattr(self, "property_states")

        if not isinstance(material, str):
            raise TypeError(
                f"{self.name}: MaterialLookup only accepts a material name "
                f"or material alias string. Got {type(material).__name__}."
            )

        self._material_input = material
        self._material_name = Material._normalize_name(material)

        self.material = self._material_name
        self.allow_extrapolation = bool(allow_extrapolation)

        self._Material = None
        self._last_temperature: float | None = None
        self._property_cache: dict[str, float] = {}

        self._property_states: dict[str, State] = {}
        self._external_property_names: set[str] = set()

        # Temperature is the only material input.
        # If none is provided, use a normal assignable default State.
        if _input_map["temperature"] is None:
            self.temperature = State(293.15)

        self._initialize_backend()

        # Ensure temperature is always an assignable State.
        state = getattr(self, "temperature", None)

        if hasattr(state, "is_assigned"):
            if self._Material is not None and not state.is_assigned:
                state.value = self._Material.temperature
        else:
            self.temperature = State(state)

        for prop_name, state in property_states.items():

            state = self.initialize_attribute(state)

            if not isinstance(state, State):
                raise TypeError(
                    f"{prop_name!r} must be a State, "
                    f"got {type(state).__name__}."
                )

            if prop_name == "temperature":
                raise ValueError(
                    "'temperature' is already being used as the material input "
                    "and cannot also be used as an output property State."
                )

            if not self._is_material_property(prop_name):
                raise AttributeError(
                    f"{prop_name!r} is not a valid Material property."
                )

            self._property_states[prop_name] = state
            self._external_property_names.add(prop_name)

    @property
    def Material(self) -> Material:
        return self._Material

    @property
    def material_name(self) -> str:
        """Return the canonical ThermoProp material name."""
        return self._material_name

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self) -> None:
        try:
            self._set_material_from_temperature()

        except Exception as e:
            raise InvalidMaterialPropertyError(
                f"{self.name}: invalid material state.\n"
                f"Material: {self._material_name!r}\n"
                f"Temperature: {self.temperature.value}"
            ) from e

        for prop_name in self._external_property_names:
            self._property_states[prop_name].value = self._get_cached_property(
                prop_name
            )

    def __getattr__(self, name: str) -> State:

        if "_Material" not in self.__dict__:
            raise AttributeError(name)

        if not self._is_material_property(name):
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        if name not in self._property_states:
            self._property_states[name] = State._derived(
                lambda prop=name: self._get_cached_property(prop)
            )

        return self._property_states[name]

    def _initialize_backend(self) -> None:
        """Create the ThermoProp Material object."""
        self._Material = Material(
            self._material_name,
            temperature=self.temperature.value,
            allow_extrapolation=self.allow_extrapolation,
        )

        self._last_temperature = None
        self._property_cache.clear()

    def _set_material_from_temperature(self) -> None:

        temperature = self.temperature.value

        if self._temperature_unchanged(temperature):
            return

        self._Material.temperature = temperature
        self._Material.allow_extrapolation = self.allow_extrapolation

        self._last_temperature = temperature
        self._property_cache.clear()

    def _temperature_unchanged(
        self,
        temperature: float,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> bool:

        if self._last_temperature is None:
            return False

        return np.isclose(
            temperature,
            self._last_temperature,
            rtol=rtol,
            atol=atol,
        )

    def _get_cached_property(self, name: str):

        if self._Material is None:
            raise ValueError(
                f"{self.name}: cannot evaluate {name!r} because the "
                "material backend is not initialized."
            )

        if name not in self._property_cache:
            self._property_cache[name] = getattr(self._Material, name)

        return self._property_cache[name]

    def _is_material_property(self, name: str) -> bool:
        return isinstance(
            getattr(Material, name, None),
            property,
        )
        
    @classmethod
    def supported_properties(cls) -> list[str]:
        return Material.supported_properties()


    @classmethod
    def show_supported_properties(cls) -> list[str]:
        return Material.show_supported_properties()


    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        return Material.supports_property(property_name)
        
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
            "Material",
            "_Material",
            "input_map",
            "_input_map",
            "material_input",
            "_material_input",
            "material_name",
            "_material_name",
            "last_temperature",
            "_last_temperature",
            "property_cache",
            "_property_cache",
        }