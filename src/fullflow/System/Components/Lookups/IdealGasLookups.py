from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State, Composition
from thermoprop import Fluid, IdealGas

from fullflow.Exceptions import InvalidThermoStateError

if TYPE_CHECKING:
    from fullflow.System import Network


FluidInput = str | dict[str, State | float] | Composition

class IdealGasLookup(Component):
    """
    ThermoProp IdealGas-backed thermodynamic property lookup component.

    `IdealGasLookup` evaluates ideal-gas thermodynamic and transport properties
    using the ThermoProp `IdealGas` wrapper. The component supports pure species
    and mixtures, automatically performs thermodynamic flash calculations from
    specified input states, and writes requested output properties to their
    corresponding states.

    Additional supported ideal-gas properties can be accessed lazily as derived
    states.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    fluid : str or dict[str, State or float] or Composition
        Ideal-gas species or composition
    pressure : State or float, optional
        Pressure flash input
    temperature : State or float, optional
        Temperature flash input
    enthalpy : State or float, optional
        Enthalpy flash input
    internal_energy : State or float, optional
        Internal energy flash input
    density : State or float, optional
        Density flash input
    flash_values : tuple[str, ...], optional
        Explicit flash input selection
    adjust_reference : bool, optional
        Whether thermodynamic reference-state offsets are applied
    **property_states : State
        Additional requested IdealGas property output states

    Outputs
    -------
    property_states : State
        Requested thermodynamic and transport property states

    Notes
    -----
    PYroMat and CoolProp use different thermodynamic reference states for
    enthalpy, internal energy, entropy, Gibbs free energy, and Helmholtz
    free energy.

    When `adjust_reference=True`, constant reference-state shifts are applied
    so that these properties are reported on the same basis as the equivalent
    CoolProp fluid at the reference state.

    Adjusted properties include:

    * `enthalpy`
    * `internal_energy`
    * `entropy`
    * `gibbs_energy`
    * `free_energy`
    * `helmholtz_energy`

    Thermodynamic differences and derivatives are preserved while arbitrary
    backend-specific reference-state offsets are removed.

    Properties derived from temperature, pressure, density, transport
    correlations, or thermodynamic derivatives are not modified.

    Flash calculations may be specified explicitly:

        ``flash_values=("pressure", "temperature")``

        ``flash_values=("pressure", "enthalpy")``

        ``flash_values=("density", "temperature")``

    Single-input flashes are supported for:

        ``("temperature",)``

        ``("enthalpy",)``

        ``("internal_energy",)``

    If `flash_values` is omitted, the component automatically selects flash
    variables from the provided thermodynamic inputs.

    Mixtures may be specified using a composition dictionary:

        ``{"N2": 0.79, "O2": 0.21}``

    or a `Composition` object.

    Composition mass fractions must sum to unity.

    Supported properties can be inspected with:

        ``IdealGasLookup.supported_properties()``

        ``IdealGasLookup.show_supported_properties()``

        ``IdealGasLookup.supports_property(property_name)``

    Supported inputs can be inspected with:

        ``IdealGasLookup.supported_inputs()``

        ``IdealGasLookup.show_supported_inputs()``

    Supported flash pairs can be inspected with:

        ``IdealGasLookup.supported_flash_pairs()``

        ``IdealGasLookup.show_supported_flash_pairs()``

    If the requested thermodynamic state is invalid,
    `InvalidThermoStateError` is raised with the active flash variables
    and composition.
    """
    _REFERENCE_TEMPERATURE = 298.15
    _REFERENCE_PRESSURE = 101325.0

    _THERMO_NAMES = (
        "pressure",
        "temperature",
        "enthalpy",
        "internal_energy",
        "density",
    )

    _PRESSURE_REQUIRED_PROPERTIES = {
        "density",
        "specific_volume",
        "entropy",
        "free_energy",
        "helmholtz_energy",
        "gibbs_energy",
    }

    _FLASH_PAIR_SETTERS = {
        frozenset(("pressure", "temperature")): ("pressure_temperature", ("pressure", "temperature")),
        frozenset(("pressure", "enthalpy")): ("pressure_enthalpy", ("pressure", "enthalpy")),
        frozenset(("pressure", "internal_energy")): ("pressure_internal_energy", ("pressure", "internal_energy")),
        frozenset(("pressure", "density")): ("pressure_density", ("pressure", "density")),
        frozenset(("density", "temperature")): ("density_temperature", ("density", "temperature")),
        frozenset(("density", "enthalpy")): ("density_enthalpy", ("density", "enthalpy")),
        frozenset(("density", "internal_energy")): ("density_internal_energy", ("density", "internal_energy")),
    }

    _SINGLE_FLASH_NAMES = {
        "temperature",
        "enthalpy",
        "internal_energy",
    }

    def __init__(
        self,
        name: str,
        network: Network,
        fluid: FluidInput,
        pressure: State | float | None = None,
        temperature: State | float | None = None,
        enthalpy: State | float | None = None,
        internal_energy: State | float | None = None,
        density: State | float | None = None,
        flash_values: tuple[str, ...] | None = None,
        adjust_reference: bool = True,
        **property_states: State,
    ):

        _input_map = {
            "pressure": pressure,
            "temperature": temperature,
            "enthalpy": enthalpy,
            "internal_energy": internal_energy,
            "density": density,
        }

        self.setup()

        if hasattr(self, "_input_map"):
            delattr(self, "_input_map")

        if hasattr(self, "property_states"):
            delattr(self, "property_states")

        self.adjust_reference = bool(adjust_reference)

        initial_fluid = self.fluid
        self.composition = self._initialize_composition(initial_fluid)
        self.fluid = self.composition

        self._last_composition_values: tuple[float, ...] | None = None

        self._coolprop_fluid = None
        self._pyromat_fluid = None

        self._reference_enthalpy = None
        self._reference_internal_energy = None
        self._reference_entropy = None
        self._reference_IdealGas = None
        self._IdealGas = None

        provided_names = [
            prop_name
            for prop_name in self._THERMO_NAMES
            if _input_map[prop_name] is not None
        ]

        if len(provided_names) == 0:
            raise ValueError(
                "IdealGasLookup requires at least one thermodynamic input "
                "so the initial ideal-gas state can be defined."
            )

        if flash_values is None:
            if "pressure" in provided_names:
                if len(provided_names) < 2:
                    raise ValueError(
                        "pressure cannot define an ideal-gas state by itself."
                    )

                self._flash_names = provided_names[:2]

                if "pressure" not in self._flash_names:
                    raise ValueError(
                        "If pressure is provided, it must be one of the first "
                        "two thermodynamic inputs."
                    )

            elif "density" in provided_names:
                if len(provided_names) < 2:
                    raise ValueError(
                        "density cannot define an ideal-gas state by itself."
                    )

                self._flash_names = provided_names[:2]

                if "density" not in self._flash_names:
                    raise ValueError(
                        "If density is provided, it must be one of the first "
                        "two thermodynamic inputs."
                    )

            else:
                self._flash_names = [provided_names[0]]

        else:
            if not isinstance(flash_values, tuple) or len(flash_values) not in {1, 2}:
                raise ValueError(
                    "flash_values must be None, a tuple with one property name, "
                    "or a tuple with two property names. Examples: "
                    "('temperature',) or ('pressure', 'enthalpy')."
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

        self._validate_flash_names()

        if len(provided_names) == 1:
            initial_flash_names = [provided_names[0]]
        else:
            initial_flash_names = provided_names[:2]

        self._validate_initial_flash_names(initial_flash_names)
        self._initial_flash_names = initial_flash_names

        self._last_flash_values: tuple[float, ...] | None = None
        self._property_cache: dict[str, float] = {}

        self._property_states: dict[str, State] = {}
        self._external_property_names: set[str] = set()

        if self.composition.is_assigned and self._composition_is_valid():
            self._initialize_backend()

        for flash_name in self._flash_names:
            state = getattr(self, flash_name, None)

            if hasattr(state, "is_assigned"):
                if self._IdealGas is not None and not state.is_assigned:
                    state.value = self._get_property(flash_name)
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
                    f"{prop_name!r} is already being used as a flash input and "
                    f"cannot also be used as an output property State."
                )

            if not self._is_ideal_gas_property(prop_name):
                raise AttributeError(
                    f"{prop_name!r} is not a valid IdealGas property."
                )

            self._property_states[prop_name] = state
            self._external_property_names.add(prop_name)

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self) -> None:
        if not self._ensure_backend_initialized():
            return

        self._set_ideal_gas_from_composition()

        try:
            self._set_ideal_gas_from_flash()

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
                f"{self.name}: invalid ideal-gas state.\n"
                f"Flash variables: {flash_state}\n"
                f"Composition: {composition}"
            ) from e

        for prop_name in self._external_property_names:
            self._property_states[prop_name].value = self._get_property(
                prop_name
            )

    def __getattr__(self, name: str) -> State:

        if "_IdealGas" not in self.__dict__:
            raise AttributeError(name)

        if not self._is_ideal_gas_property(name):
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        if name not in self._property_states:
            self._property_states[name] = State._derived(
                lambda prop=name: self._get_property(prop)
            )

        return self._property_states[name]

    def _initialize_backend(self) -> None:
        """Initialize PYroMat and, if requested, the matching Fluid reference."""
        self._coolprop_fluid = self._coolprop_argument_from_composition()
        self._pyromat_fluid = self._pyromat_argument_from_composition()

        self._reference_IdealGas = IdealGas(
            self._pyromat_fluid,
            basis="mass",
            pressure=self._REFERENCE_PRESSURE,
            temperature=self._REFERENCE_TEMPERATURE,
        )

        self._update_reference_properties()

        self._IdealGas = IdealGas(
            self._pyromat_fluid,
            basis="mass",
            **{
                flash_name: self._to_ideal_basis(
                    flash_name,
                    getattr(self, flash_name).value,
                )
                for flash_name in self._initial_flash_names
            },
        )

        self._last_flash_values = None
        self._property_cache.clear()

    def _update_reference_properties(self) -> None:
        """
        Store Fluid reference properties for PYroMat-to-Fluid shifts.

        If adjust_reference is False, this intentionally does nothing and all
        ideal-gas properties remain on the raw PYroMat reference basis.
        """
        self._reference_enthalpy = None
        self._reference_internal_energy = None
        self._reference_entropy = None

        if not self.adjust_reference:
            return

        try:
            reference_fluid = Fluid(
                self._coolprop_fluid,
                basis="mass",
                pressure=self._REFERENCE_PRESSURE,
                temperature=self._REFERENCE_TEMPERATURE,
            )

        except Exception as e:
            composition = (
                self._coolprop_fluid
                if isinstance(self._coolprop_fluid, str)
                else dict(self._coolprop_fluid)
            )

            raise ValueError(
                f"{self.name}: adjust_reference=True requires the ideal-gas "
                "species/composition to also be available through the "
                "CoolProp-backed Fluid wrapper.\n"
                f"Composition attempted for Fluid reference: {composition!r}\n"
                "Set adjust_reference=False to use raw PYroMat reference values "
                "instead."
            ) from e

        self._reference_enthalpy = reference_fluid.enthalpy
        self._reference_internal_energy = reference_fluid.internal_energy
        self._reference_entropy = reference_fluid.entropy

    def _ensure_backend_initialized(self) -> bool:
        if self._IdealGas is not None:
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

    def _validate_initial_flash_names(self, flash_names: list[str]) -> None:
        if len(flash_names) == 1:
            if flash_names[0] not in self._SINGLE_FLASH_NAMES:
                raise ValueError(
                    f"Unsupported initial IdealGas flash input: {flash_names[0]!r}."
                )
            return

        key = frozenset(flash_names)

        if key not in self._FLASH_PAIR_SETTERS:
            raise ValueError(
                f"Unsupported initial IdealGas flash pair: {sorted(flash_names)}."
            )

    def _validate_flash_names(self) -> None:
        if len(self._flash_names) == 1:
            if self._flash_names[0] not in self._SINGLE_FLASH_NAMES:
                raise ValueError(
                    f"Unsupported IdealGas flash input: {self._flash_names[0]!r}."
                )
            return

        key = frozenset(self._flash_names)

        if key not in self._FLASH_PAIR_SETTERS:
            raise ValueError(
                f"Unsupported IdealGas flash pair: {sorted(self._flash_names)}."
            )

    def _set_ideal_gas_from_flash(self) -> None:
        if len(self._flash_names) == 1:
            flash_name = self._flash_names[0]

            flash_values = (
                self._to_ideal_basis(
                    flash_name,
                    getattr(self, flash_name).value,
                ),
            )

            if self._flash_values_unchanged(flash_values):
                return

            setattr(
                self._IdealGas,
                flash_name,
                flash_values[0],
            )

            self._last_flash_values = flash_values
            self._property_cache.clear()

            return

        setter_name, ordered_names = self._FLASH_PAIR_SETTERS[
            frozenset(self._flash_names)
        ]

        flash_values = tuple(
            self._to_ideal_basis(
                prop_name,
                getattr(self, prop_name).value,
            )
            for prop_name in ordered_names
        )

        if self._flash_values_unchanged(flash_values):
            return

        setattr(
            self._IdealGas,
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

    def _to_ideal_basis(self, name: str, value: float) -> float:
        """Convert externally supplied Fluid-basis values to PYroMat basis."""
        if not self.adjust_reference:
            return float(value)

        if name == "enthalpy":
            return (
                self._reference_IdealGas.enthalpy
                + float(value)
                - self._reference_enthalpy
            )

        if name == "internal_energy":
            return (
                self._reference_IdealGas.internal_energy
                + float(value)
                - self._reference_internal_energy
            )

        return float(value)


    def _from_ideal_basis(self, name: str, value: float) -> float:
        """Convert PYroMat-basis properties to the Fluid reference basis."""
        if not self.adjust_reference:
            return float(value)

        enthalpy_offset = (
            self._reference_enthalpy
            - self._reference_IdealGas.enthalpy
        )

        internal_energy_offset = (
            self._reference_internal_energy
            - self._reference_IdealGas.internal_energy
        )

        entropy_offset = (
            self._reference_entropy
            - self._reference_IdealGas.entropy
        )

        if name == "enthalpy":
            return float(value) + enthalpy_offset

        if name == "internal_energy":
            return float(value) + internal_energy_offset

        if name == "entropy":
            return float(value) + entropy_offset

        if name == "gibbs_energy":
            return (
                float(value)
                + enthalpy_offset
                - self._IdealGas.temperature * entropy_offset
            )

        if name in ("free_energy", "helmholtz_energy"):
            return (
                float(value)
                + internal_energy_offset
                - self._IdealGas.temperature * entropy_offset
            )

        return float(value)



    def _get_property(self, name: str):
        if not self._ensure_backend_initialized():
            raise ValueError(
                f"{self.name}: cannot evaluate {name!r} because the "
                "ideal-gas composition is not initialized yet."
            )

        if self._requires_pressure(name) and self._IdealGas.pressure is None:
            raise ValueError(
                f"{name!r} requires pressure, but pressure is not available."
            )

        if name not in self._property_cache:
            value = getattr(self._IdealGas, name)
            self._property_cache[name] = self._from_ideal_basis(name, value)

        return self._property_cache[name]

    def _requires_pressure(self, name: str) -> bool:
        return name in self._PRESSURE_REQUIRED_PROPERTIES

    def _is_ideal_gas_property(self, name: str) -> bool:
        return isinstance(
            getattr(IdealGas, name, None),
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

    def _coolprop_argument_from_composition(self) -> str | dict[str, float]:

        values = self.composition.values

        if len(values) == 1:
            return next(iter(values))

        return dict(values)

    def _pyromat_argument_from_composition(self) -> str | dict[str, float]:

        values = self.composition.values

        if len(values) == 1:
            return next(iter(values))

        return dict(values)

    def _composition_values(self) -> tuple[float, ...]:
        return tuple(
            self.composition[species].value
            for species in self.composition.species
        )

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

    def _set_ideal_gas_from_composition(self) -> None:

        composition_values = self._composition_values()

        if self._composition_values_unchanged(composition_values):
            return

        total = sum(composition_values)

        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{self.name}: composition mass fractions must sum to 1.0. "
                f"Got {total}."
            )

        new_coolprop_fluid = self._coolprop_argument_from_composition()
        new_pyromat_fluid = self._pyromat_argument_from_composition()

        # Species set changed, so rebuild the ideal-gas and reference backends.
        if (
            new_coolprop_fluid != self._coolprop_fluid
            or new_pyromat_fluid != self._pyromat_fluid
        ):
            self._initialize_backend()

            self._last_composition_values = composition_values
            self._last_flash_values = None
            self._property_cache.clear()
            return

        # Same species set, only fractions changed.
        if len(composition_values) > 1:
            self._IdealGas.mass_fractions = list(composition_values)
            self._reference_IdealGas.mass_fractions = list(composition_values)

            self._update_reference_properties()

        self._last_composition_values = composition_values
        self._last_flash_values = None
        self._property_cache.clear()


    @classmethod
    def supported_properties(cls) -> list[str]:
        return IdealGas.supported_properties()


    @classmethod
    def show_supported_properties(cls) -> list[str]:
        return IdealGas.show_supported_properties()


    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        return IdealGas.supports_property(property_name)
        
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
        pairs = [
            "-".join(ordered_names)
            for _, ordered_names in cls._FLASH_PAIR_SETTERS.values()
        ]

        pairs.extend(cls._SINGLE_FLASH_NAMES)

        return sorted(pairs)


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
            "IdealGas",
            "_IdealGas",
            "reference_IdealGas",
            "_reference_IdealGas",
            "coolprop_fluid",
            "_coolprop_fluid",
            "pyromat_fluid",
            "_pyromat_fluid",
            "reference_enthalpy",
            "_reference_enthalpy",
            "reference_internal_energy",
            "_reference_internal_energy",
            "reference_entropy",
            "_reference_entropy",
            "adjust_reference",
            "input_map",
            "_input_map",
            "last_flash_values",
            "_last_flash_values",
            "property_cache",
            "_property_cache",
            "composition",
            "last_composition_values",
            "_last_composition_values",
            "initial_flash_names",
            "_initial_flash_names",
        }