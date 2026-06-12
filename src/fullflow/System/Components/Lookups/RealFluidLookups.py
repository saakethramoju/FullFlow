from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State, Composition
from thermoprop import Fluid

from fullflow.Exceptions import InvalidThermoStateError

if TYPE_CHECKING:
    from fullflow.System import Network


FluidInput = str | dict[str, State | float] | Composition


class FluidLookup(Component):
    """
    CoolProp-backed thermodynamic property lookup component.

    `FluidLookup` evaluates thermodynamic and transport properties through the
    ThermoProp `Fluid` wrapper. It supports pure fluids, mixtures, and dynamic
    composition inputs through `Composition`.

    The component flashes the fluid state from two selected thermodynamic
    inputs, then writes requested property outputs to their corresponding
    states. Additional supported fluid properties can be accessed lazily as
    derived states.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    fluid : str or dict or Composition
        Fluid name, species-fraction dictionary, or composition object
    pressure : State or float, optional
        Pressure input or output state
    temperature : State or float, optional
        Temperature input or output state
    enthalpy : State or float, optional
        Specific enthalpy input or output state
    quality : State or float, optional
        Vapor quality input or output state
    density : State or float, optional
        Density input or output state
    internal_energy : State or float, optional
        Specific internal energy input or output state
    flash_values : tuple[str, str], optional
        Thermodynamic property pair used for the flash
    **property_states : State
        Additional requested Fluid property output states

    Outputs
    -------
    property_states : State
        Requested thermodynamic or transport property states

    Notes
    -----
    `FluidLookup` requires at least two thermodynamic inputs so the initial
    fluid state can be defined.

    If `flash_values` is omitted, the first two provided thermodynamic inputs
    are used as the flash pair. If `flash_values` is provided, it must be a
    supported pair of thermodynamic property names.

    Supported thermodynamic inputs are:

        ``pressure, temperature, enthalpy, quality, density, internal_energy``

    Supported flash pairs are:

        ``pressure-temperature``

        ``pressure-enthalpy``

        ``pressure-quality``

        ``temperature-quality``

        ``density-internal_energy``

        ``pressure-density``

        ``pressure-internal_energy``

        ``temperature-density``

        ``density-enthalpy``

        ``temperature-enthalpy``

    Pure fluids may be passed as strings:

        ``fluid = "water"``

    Mixtures may be passed as dictionaries:

        ``fluid = {"oxygen": 0.7, "nitrogen": 0.3}``

    Dynamic mixtures may be passed as `Composition` objects. Composition mass
    fractions must sum to one before the backend can be initialized.

    Requested non-flash thermodynamic states are treated as outputs. Additional
    output properties may be passed with keyword arguments:

        ``FluidLookup(..., pressure=P, temperature=T, viscosity=mu)``

    Supported properties can be inspected with:

        ``FluidLookup.supported_properties()``

        ``FluidLookup.show_supported_properties()``

        ``FluidLookup.supports_property(property_name)``

    Supported inputs and flash pairs can be inspected with:

        ``FluidLookup.supported_inputs()``

        ``FluidLookup.show_supported_inputs()``

        ``FluidLookup.supported_flash_pairs()``

        ``FluidLookup.show_supported_flash_pairs()``

    If the thermodynamic state is invalid, `InvalidThermoStateError` is raised
    with the flash variables and composition values.
    """
    
    _THERMO_NAMES = (
        "pressure",
        "temperature",
        "enthalpy",
        "quality",
        "density",
        "internal_energy",
    )

    _FLASH_PAIR_SETTERS = {
        frozenset(("pressure", "temperature")): ("pressure_temperature", ("pressure", "temperature")),
        frozenset(("pressure", "enthalpy")): ("pressure_enthalpy", ("pressure", "enthalpy")),
        frozenset(("pressure", "quality")): ("pressure_quality", ("pressure", "quality")),
        frozenset(("temperature", "quality")): ("temperature_quality", ("temperature", "quality")),
        frozenset(("density", "internal_energy")): ("density_internal_energy", ("density", "internal_energy")),
        frozenset(("pressure", "density")): ("pressure_density", ("pressure", "density")),
        frozenset(("pressure", "internal_energy")): ("pressure_internal_energy", ("pressure", "internal_energy")),
        frozenset(("temperature", "density")): ("temperature_density", ("temperature", "density")),
        frozenset(("density", "enthalpy")): ("density_enthalpy", ("density", "enthalpy")),
        frozenset(("temperature", "enthalpy")): ("temperature_enthalpy", ("temperature", "enthalpy")),
    }

    def __init__(
        self,
        name: str,
        network: Network,
        fluid: FluidInput,
        pressure: State | float | None = None,
        temperature: State | float | None = None,
        enthalpy: State | float | None = None,
        quality: State | float | None = None,
        density: State | float | None = None,
        internal_energy: State | float | None = None,
        flash_values: tuple[str, str] | None = None,
        **property_states: State,
    ):

        _input_map = {
            "pressure": pressure,
            "temperature": temperature,
            "enthalpy": enthalpy,
            "quality": quality,
            "density": density,
            "internal_energy": internal_energy,
        }

        self.setup()

        if hasattr(self, "_input_map"):
            delattr(self, "_input_map")

        if hasattr(self, "property_states"):
            delattr(self, "property_states")

        initial_fluid = self.fluid
        self.composition = self._initialize_composition(initial_fluid)
        self.fluid = self.composition

        self._coolprop_fluid = None
        self._Fluid = None
        self._last_composition_values: tuple[float, ...] | None = None

        provided_names = [
            prop_name
            for prop_name in self._THERMO_NAMES
            if _input_map[prop_name] is not None
        ]

        if len(provided_names) < 2:
            raise ValueError(
                "FluidLookup requires at least two provided thermodynamic inputs "
                "so the initial fluid state can be defined."
            )

        if flash_values is None:
            self._flash_names = provided_names[:2]
        else:
            if not isinstance(flash_values, tuple) or len(flash_values) != 2:
                raise ValueError(
                    "flash_values must be None or a tuple of two property names, "
                    "for example ('pressure', 'enthalpy')."
                )

            self._flash_names = list(flash_values)

            invalid_flash_values = [
                name for name in self._flash_names
                if name not in self._THERMO_NAMES
            ]

            if invalid_flash_values:
                raise ValueError(
                    f"Invalid flash_values: {invalid_flash_values}. "
                    f"Valid names are: {list(self._THERMO_NAMES)}."
                )

        flash_key = frozenset(self._flash_names)

        if flash_key not in self._FLASH_PAIR_SETTERS:
            raise ValueError(
                f"Unsupported Fluid flash pair: {sorted(self._flash_names)}. "
                f"Supported pairs are: {Fluid.available_flash_pairs()}."
            )

        initial_flash_names = provided_names[:2]
        initial_flash_key = frozenset(initial_flash_names)

        if initial_flash_key not in self._FLASH_PAIR_SETTERS:
            raise ValueError(
                f"Unsupported initial Fluid flash pair: {sorted(initial_flash_names)}. "
                f"Supported pairs are: {Fluid.available_flash_pairs()}."
            )

        _, initial_ordered_names = self._FLASH_PAIR_SETTERS[initial_flash_key]
        self._initial_ordered_names = initial_ordered_names

        self._last_flash_values: tuple[float, ...] | None = None
        self._property_cache: dict[str, float] = {}

        self._property_states: dict[str, State] = {}
        self._external_property_names: set[str] = set()

        # Initialize now if possible; otherwise defer until composition is valid.
        if self.composition.is_assigned and self._composition_is_valid():
            self._initialize_backend()

        # Flash states should be assignable States.
        for flash_name in self._flash_names:
            state = getattr(self, flash_name, None)

            if hasattr(state, "is_assigned"):
                if self._Fluid is not None and not state.is_assigned:
                    state.value = self._get_cached_property(flash_name)
            else:
                setattr(self, flash_name, State(state))

        # Remove unused placeholder thermo states.
        for prop_name in self._THERMO_NAMES:
            if prop_name in self._flash_names:
                continue

            if _input_map[prop_name] is None and prop_name in self.__dict__:
                delattr(self, prop_name)

        # Provided non-flash thermo states become output states.
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
                    f"{prop_name!r} is already being used as a flash input and "
                    f"cannot also be used as an output property State."
                )

            if not self._is_fluid_property(prop_name):
                raise AttributeError(
                    f"{prop_name!r} is not a valid Fluid property."
                )

            self._property_states[prop_name] = state
            self._external_property_names.add(prop_name)

    def pre_evaluation(self):
        self.evaluate_states()


    def evaluate_states(self) -> None:
        if not self._ensure_backend_initialized():
            return

        self._set_fluid_from_composition()

        try:
            self._set_fluid_from_flash()

        except Exception as e:
            flash_state = {
                name: getattr(self, name).value
                for name in self._flash_names
            }

            composition = {
                species: state.value
                for species, state in self.composition
            }

            raise InvalidThermoStateError(
                f"{self.name}: invalid fluid state.\n"
                f"Flash variables: {flash_state}\n"
                f"Composition: {composition}"
            ) from e

        for prop_name in self._external_property_names:
            self._property_states[prop_name].value = self._get_cached_property(
                prop_name
            )


            
    def __getattr__(self, name: str) -> State:

        if "_Fluid" not in self.__dict__:
            raise AttributeError(name)

        if not self._is_fluid_property(name):
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        if name not in self._property_states:
            self._property_states[name] = State._derived(
                lambda prop=name: self._get_cached_property(prop)
            )

        return self._property_states[name]

    def _initialize_backend(self) -> None:
        self._coolprop_fluid = self._fluid_argument_from_composition()

        self._Fluid = Fluid(
            self._coolprop_fluid,
            basis="mass",
            **{
                name: getattr(self, name).value
                for name in self._initial_ordered_names
            },
        )

        self._sync_composition_species_from_backend()
        self._coolprop_fluid = self._fluid_argument_from_composition()

        self._last_flash_values = None
        self._property_cache.clear()

    def _sync_composition_species_from_backend(self) -> None:
        """Sync composition keys to the ThermoProp Fluid species names."""
        if self._Fluid is None or not self.composition.is_assigned:
            return

        species = tuple(getattr(self._Fluid, "species", ()))

        if not species:
            return

        self.composition.sync_species(species)

    def _ensure_backend_initialized(self) -> bool:
        if self._Fluid is not None:
            return True

        if not self.composition.is_assigned:
            return False

        if not self._composition_is_valid():
            return False

        self._initialize_backend()
        return True

    def _composition_is_valid(self) -> bool:
        if not self.composition.is_assigned:
            return False

        values = tuple(
            self.composition[species].value
            for species in self.composition.species
        )

        return np.isclose(sum(values), 1.0, rtol=0.0, atol=1e-6)

    def _set_fluid_from_flash(self) -> None:

        setter_name, ordered_names = self._FLASH_PAIR_SETTERS[
            frozenset(self._flash_names)
        ]

        flash_values = tuple(
            getattr(self, prop_name).value
            for prop_name in ordered_names
        )

        if self._flash_values_unchanged(flash_values):
            return

        setattr(
            self._Fluid,
            setter_name,
            flash_values,
        )

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

        if not self._ensure_backend_initialized():
            raise ValueError(
                f"{self.name}: cannot evaluate {name!r} because the "
                "fluid composition is not initialized yet."
            )

        if name not in self._property_cache:
            self._property_cache[name] = getattr(self._Fluid, name)

        return self._property_cache[name]

    def _is_fluid_property(self, name: str) -> bool:
        return isinstance(
            getattr(Fluid, name, None),
            property,
        )

    def _initialize_composition(self, fluid: FluidInput) -> Composition:

        if isinstance(fluid, Composition):
            return fluid

        try:
            composition = Composition(fluid)
        except Exception as e:
            raise ValueError(
                f"{self.name}: invalid fluid input {fluid!r}. "
                "Expected a fluid name, a species-fraction dictionary, "
                "or a Composition object."
            ) from e

        if not composition.is_assigned:
            raise ValueError(
                f"{self.name}: composition must contain at least one species."
            )

        return composition

    def _fluid_argument_from_composition(self) -> str | dict[str, float]:

        values = self.composition.values

        if len(values) == 1:
            return next(iter(values))

        return dict(values)

    def _composition_values(self) -> tuple[float, ...]:
        return tuple(
            self.composition[species].value
            for species in self.composition.species
        )

    def _set_fluid_from_composition(self) -> None:

        composition_values = self._composition_values()

        if self._composition_values_unchanged(composition_values):
            return

        total = sum(composition_values)

        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{self.name}: composition mass fractions must sum to 1.0. "
                f"Got {total}."
            )

        new_coolprop_fluid = self._fluid_argument_from_composition()

        # Species set changed, so rebuild the CoolProp backend.
        if new_coolprop_fluid != self._coolprop_fluid:
            self._coolprop_fluid = new_coolprop_fluid

            self._Fluid = Fluid(
                self._coolprop_fluid,
                basis="mass",
                **{
                    name: getattr(self, name).value
                    for name in self._initial_ordered_names
                },
            )

            self._sync_composition_species_from_backend()
            self._coolprop_fluid = self._fluid_argument_from_composition()
            composition_values = self._composition_values()

            self._last_composition_values = composition_values
            self._last_flash_values = None
            self._property_cache.clear()
            return

        # Same species set, only fractions changed.
        if len(composition_values) > 1:
            self._Fluid.mass_fractions = list(composition_values)

        self._last_composition_values = composition_values
        self._last_flash_values = None
        self._property_cache.clear()


    def _composition_values_unchanged(
        self,
        composition_values: tuple[float, ...],
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> bool:

        if self._last_composition_values is None:
            return False

        return all(
            np.isclose(current, previous, rtol=rtol, atol=atol)
            for current, previous in zip(
                composition_values,
                self._last_composition_values,
            )
        )
        
    @classmethod
    def supported_properties(cls) -> list[str]:
        return Fluid.supported_properties()


    @classmethod
    def show_supported_properties(cls) -> list[str]:
        return Fluid.show_supported_properties()


    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        return Fluid.supports_property(property_name)
    

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
        return sorted(
            "-".join(ordered_names)
            for _, ordered_names in cls._FLASH_PAIR_SETTERS.values()
        )


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
            "Fluid",
            "_Fluid",
            "input_map",
            "_input_map",
            "coolprop_fluid",
            "_coolprop_fluid",
            "last_flash_values",
            "_last_flash_values",
            "property_cache",
            "_property_cache",
            "composition",
            "last_composition_values",
            "_last_composition_values",
            "initial_ordered_names",
            "_initial_ordered_names",
        }