from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from fullflow.System import Component, State, Composition
from fullflow.Exceptions import InvalidThermoStateError
from thermoprop import Equilibrium, Reactants, CombustionGas, Propellant

if TYPE_CHECKING:
    from fullflow.System import Network


CombustionInput = Reactants | CombustionGas | dict[str, State | float] | Composition | State

class CombustionLookup(Component):
    """
    ThermoProp Equilibrium-backed combustion property lookup component.

    `CombustionLookup` evaluates equilibrium combustion-gas properties through
    the ThermoProp `Equilibrium` wrapper. The lookup supports HP and TP modes.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    reactants : Reactants, CombustionGas, dict, Composition, or State
        Reactants, combustion-gas object, species mass-fraction dictionary,
        Composition object, or State containing one of these objects passed to
        `Equilibrium`. Dictionary and Composition inputs are converted to a
        ThermoProp `CombustionGas`.
    mode : str, optional
        Equilibrium mode
    pressure : State or float, optional
        Pressure input [Pa]
    temperature : State or float, optional
        Temperature input [K] for TP mode. In HP mode this may be used as an
        initial temperature guess for Reactants inputs. For dict or Composition
        inputs, temperature is required because the input is converted to a
        `CombustionGas`.
    flash_values : tuple[str, ...], optional
        Explicit flash input selection
    CombustionGas_trace : float, optional
        Minimum mole fraction retained when constructing the output
        `CombustionGas` object.
    CombustionGas_max_species : int or None, optional
        Maximum number of species retained when constructing the output
        `CombustionGas` object.
    guess_temperature : float, optional
        Initial temperature guess [K] for HP equilibrium solves.
    candidates : list[str] or None, optional
        User-supplied list of allowable equilibrium product species.
    include_all_valid_gases : bool, optional
        If True, all valid gas-phase CEA species consistent with the reactant
        elemental inventory are considered.
    element_tol : float, optional
        Element balance convergence tolerance.
    enthalpy_tol : float, optional
        HP enthalpy convergence tolerance [J/kg].
    correction_tol : float, optional
        Newton correction convergence tolerance.
    max_iterations : int, optional
        Maximum equilibrium iterations.
    trace_moles : float, optional
        Minimum species mole count used internally for numerical stability.
    min_temperature : float, optional
        Minimum allowable equilibrium temperature [K].
    max_temperature : float, optional
        Maximum allowable equilibrium temperature [K].
    equilibrium_derivative_temperature_step : float, optional
        Temperature step [K] used when evaluating equilibrium derivative
        properties such as equilibrium specific heat.
    **property_states : State
        Additional requested Equilibrium property output states.

    Outputs
    -------
    composition : Composition
        Equilibrium product mass-fraction composition.
    CombustionGas : CombustionGas
        Equilibrium combustion-gas object returned by `Equilibrium`.
    property_states : State
        Requested equilibrium combustion-gas property states.

    Notes
    -----
    HP mode uses pressure as the flash input:

        ``flash_values = ("pressure",)``

    TP mode uses pressure and temperature as flash inputs:

        ``flash_values = ("pressure", "temperature")``

    HP equilibrium is evaluated from reactant enthalpy and pressure:

        ``gas = Equilibrium(reactants, mode="hp", pressure=pressure)``

    TP equilibrium is evaluated from temperature and pressure:

        ``gas = Equilibrium(reactants, mode="tp", pressure=pressure, temperature=temperature)``

    Dictionary and Composition inputs are converted to a `CombustionGas` using
    mass fractions:

        ``reactants = CombustionGas(composition, basis="mass", pressure=pressure, temperature=temperature)``

    The `composition` attribute is updated from:

        ``Equilibrium.CombustionGas.mass_fractions``

    Supported properties may be requested as output `State` objects or
    accessed directly through the lookup using derived states.
    """
    _THERMO_NAMES = (
        "pressure",
        "temperature",
    )

    _SUPPORTED_MODES = {
        "hp": ("pressure",),
        "tp": ("pressure", "temperature"),
    }

    _COMPOSITION_NEGATIVE_TOLERANCE = 1e-4

    def __init__(
        self,
        name: str,
        network: Network,
        reactants: CombustionInput,
        mode: str = "hp",
        pressure: State | float | None = None,
        temperature: State | float | None = None,
        flash_values: tuple[str, ...] | None = None,
        CombustionGas_trace: float = 1e-8,
        CombustionGas_max_species: int | None = 25,
        guess_temperature: float = 3500.0,
        candidates: list[str] | None = None,
        include_all_valid_gases: bool = True,
        element_tol: float = 1e-8,
        enthalpy_tol: float = 1e-3,
        correction_tol: float = 1e-8,
        max_iterations: int = 200,
        trace_moles: float = 1e-300,
        min_temperature: float = 200.0,
        max_temperature: float = 20000.0,
        equilibrium_derivative_temperature_step: float = 1.0,
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

        self._reactant_composition = self._initialize_reactant_composition(reactants)
        self._input = reactants

        if self._reactant_composition is None:
            self.reactants = reactants
        else:
            self.reactants = self._reactant_composition

        if flash_values is None:
            self._flash_names = list(self._SUPPORTED_MODES[self._mode])
        else:
            if not isinstance(flash_values, tuple):
                raise ValueError("flash_values must be a tuple of property names.")

            self._flash_names = list(flash_values)

            if tuple(self._flash_names) != self._SUPPORTED_MODES[self._mode]:
                raise ValueError(
                    f"{self._mode.upper()} mode requires "
                    f"flash_values={self._SUPPORTED_MODES[self._mode]!r}."
                )

        if "pressure" not in self._flash_names:
            raise ValueError("CombustionLookup requires pressure.")

        if _input_map["pressure"] is None:
            raise ValueError("CombustionLookup requires pressure.")

        if self._mode == "tp" and _input_map["temperature"] is None:
            raise ValueError("TP CombustionLookup requires temperature and pressure.")

        self._guess_temperature = float(guess_temperature)
        self._candidates = candidates
        self._include_all_valid_gases = bool(include_all_valid_gases)
        self._element_tol = float(element_tol)
        self._enthalpy_tol = float(enthalpy_tol)
        self._correction_tol = float(correction_tol)
        self._max_iterations = int(max_iterations)
        self._trace_moles = float(trace_moles)
        self._min_temperature = float(min_temperature)
        self._max_temperature = float(max_temperature)
        self._CombustionGas_trace = float(CombustionGas_trace)
        self._CombustionGas_max_species = CombustionGas_max_species
        self._equilibrium_derivative_temperature_step = float(equilibrium_derivative_temperature_step)

        self.CombustionGas_trace = self._CombustionGas_trace
        self.CombustionGas_max_species = self._CombustionGas_max_species

        self._Equilibrium = None
        self._last_flash_values: tuple[float, ...] | None = None
        self._last_reactant_composition_values: tuple[float, ...] | None = None
        self._property_cache: dict[str, object] = {}

        self._property_states: dict[str, State] = {}
        self._external_property_names: set[str] = set()

        self.composition = Composition()

        self._initialize_backend()
        self._sync_composition_from_equilibrium()

        for flash_name in self._flash_names:
            state = getattr(self, flash_name, None)

            if hasattr(state, "is_assigned"):
                if self._Equilibrium is not None and not state.is_assigned:
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
                    f"{prop_name!r} must be a State, got {type(state).__name__}."
                )

            if prop_name in self._flash_names:
                raise ValueError(
                    f"{prop_name!r} is already being used as a combustion "
                    f"flash input and cannot also be used as an output property State."
                )

            if not self._is_equilibrium_property(prop_name):
                raise AttributeError(
                    f"{prop_name!r} is not a valid Equilibrium property."
                )

            self._property_states[prop_name] = state
            self._external_property_names.add(prop_name)

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self) -> None:
        try:
            self._set_equilibrium_from_flash()

        except Exception as e:
            flash_state = {
                name: getattr(self, name).value
                for name in self._flash_names
            }

            raise InvalidThermoStateError(
                f"{self.name}: invalid combustion equilibrium state.\n"
                f"Mode: {self._mode!r}\n"
                f"Flash variables: {flash_state}"
            ) from e

        self._sync_composition_from_equilibrium()

        for prop_name in self._external_property_names:
            self._property_states[prop_name].value = self._get_cached_property(prop_name)

    @property
    def CombustionGas(self) -> CombustionGas:
        return self._Equilibrium.CombustionGas

    def __getattr__(self, name: str) -> State:
        if "_Equilibrium" not in self.__dict__:
            raise AttributeError(name)

        if not self._is_equilibrium_property(name):
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        if name not in self._property_states:
            self._property_states[name] = State._derived(
                lambda prop=name: self._get_cached_property(prop)
            )

        return self._property_states[name]

    def _initialize_backend(self) -> None:
        kwargs = self._equilibrium_kwargs()

        if self._mode == "hp":
            kwargs["pressure"] = self.pressure.value

            if hasattr(self, "temperature") and self.temperature.is_assigned:
                kwargs["guess_temperature"] = self.temperature.value

        elif self._mode == "tp":
            kwargs["pressure"] = self.pressure.value
            kwargs["temperature"] = self.temperature.value

        self._input = self._current_input()

        self._Equilibrium = Equilibrium(
            self._input,
            mode=self._mode,
            **kwargs,
        )

        self._last_flash_values = None
        self._property_cache.clear()

    def _set_equilibrium_from_flash(self) -> None:
        current_input = self._current_input()

        if current_input is not self._input:
            self._input = current_input
            self._initialize_backend()
            return

        flash_values = tuple(
            getattr(self, prop_name).value
            for prop_name in self._flash_names
        )

        if self._flash_values_unchanged(flash_values):
            return

        if self._Equilibrium is None:
            self._initialize_backend()
            return

        if self._mode == "hp":
            self._Equilibrium.pressure = flash_values[0]

        elif self._mode == "tp":
            self._Equilibrium.pressure_temperature = (flash_values[0], flash_values[1])

        self._last_flash_values = flash_values
        self._property_cache.clear()

    def _sync_composition_from_equilibrium(self) -> None:
        mass_fractions = dict(self._Equilibrium.CombustionGas.mass_fractions)

        if not mass_fractions:
            return

        if not self.composition.is_assigned:
            self.composition = Composition(mass_fractions)
            return

        old_species = set(self.composition.species)
        new_species = set(mass_fractions)

        if old_species != new_species:
            self.composition = Composition(mass_fractions)
            return

        for species, value in mass_fractions.items():
            self.composition[species].value = value

        self.composition.validate()

    def _equilibrium_kwargs(self) -> dict:
        return {
            "guess_temperature": self._guess_temperature,
            "candidates": self._candidates,
            "include_all_valid_gases": self._include_all_valid_gases,
            "element_tol": self._element_tol,
            "enthalpy_tol": self._enthalpy_tol,
            "correction_tol": self._correction_tol,
            "max_iterations": self._max_iterations,
            "trace_moles": self._trace_moles,
            "min_temperature": self._min_temperature,
            "max_temperature": self._max_temperature,
            "CombustionGas_trace": self._CombustionGas_trace,
            "CombustionGas_max_species": self._CombustionGas_max_species,
            "equilibrium_derivative_temperature_step": self._equilibrium_derivative_temperature_step,
        }

    def _initialize_reactant_composition(
        self,
        reactants: CombustionInput,
    ) -> Composition | None:

        if isinstance(reactants, State):
            if not reactants.is_assigned:
                return None

            reactants = reactants.value

        if isinstance(reactants, (Reactants, CombustionGas)):
            return None

        if isinstance(reactants, Composition):
            return reactants

        try:
            composition = Composition(reactants)
        except Exception as e:
            raise ValueError(
                f"{self.name}: invalid combustion input {reactants!r}. "
                "Expected Reactants, CombustionGas, species-fraction dictionary, "
                "Composition, or State containing one of these."
            ) from e

        if not composition.is_assigned:
            raise ValueError(
                f"{self.name}: composition must contain at least one species."
            )

        return composition

    def _current_input(self) -> Reactants | CombustionGas:
        current = self.reactants

        if isinstance(current, State):
            current = current.value

        if isinstance(current, (Reactants, CombustionGas)):
            return current

        if isinstance(current, Composition):
            return self._CombustionGas_from_composition(current)

        raise TypeError(
            f"{self.name}: invalid combustion input type "
            f"{type(current).__name__!r}."
        )

    def _CombustionGas_from_composition(
        self,
        composition: Composition,
    ) -> CombustionGas:

        if not composition.is_assigned:
            raise ValueError(
                f"{self.name}: composition must contain at least one species."
            )

        if not hasattr(self, "temperature") or not self.temperature.is_assigned:
            raise ValueError(
                f"{self.name}: temperature is required when reactants is a "
                "dict or Composition because it must be converted to CombustionGas."
            )

        composition_values = self._reactant_composition_values(composition)
        property_composition_values = self._property_safe_reactant_composition_values(
            composition,
            composition_values,
        )

        return CombustionGas(
            self._reactant_composition_argument_from_values(
                composition,
                property_composition_values,
            ),
            basis="mass",
            pressure=self.pressure.value,
            temperature=self.temperature.value,
        )

    def _reactant_composition_values(
        self,
        composition: Composition,
    ) -> tuple[float, ...]:

        return tuple(
            composition[species].value
            for species in composition.species
        )

    def _reactant_composition_argument_from_values(
        self,
        composition: Composition,
        composition_values: tuple[float, ...],
    ) -> dict[str, float]:

        return {
            name: value
            for name, value in zip(composition.species, composition_values)
        }

    def _property_safe_reactant_composition_values(
        self,
        composition: Composition,
        composition_values: tuple[float, ...],
    ) -> tuple[float, ...]:

        total = sum(composition_values)

        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{self.name}: composition mass fractions must sum to 1.0. "
                f"Got {total}."
            )

        for species, value in zip(composition.species, composition_values):
            if value < -self._COMPOSITION_NEGATIVE_TOLERANCE:
                raise ValueError(
                    f"{self.name}: composition contains a significantly "
                    f"negative mass fraction for {species!r}: {value}."
                )

        safe_values = tuple(
            max(0.0, float(value))
            for value in composition_values
        )

        total = sum(safe_values)

        if total <= 0.0:
            raise ValueError(
                f"{self.name}: composition has no positive mass fractions."
            )

        return tuple(
            value / total
            for value in safe_values
        )

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
            for current, previous in zip(flash_values, self._last_flash_values)
        )

    def _get_cached_property(self, name: str):
        if self._Equilibrium is None:
            raise ValueError(
                f"{self.name}: cannot evaluate {name!r} because the "
                "Equilibrium backend is not initialized."
            )

        if name not in self._property_cache:
            self._property_cache[name] = getattr(self._Equilibrium, name)

        return self._property_cache[name]

    def _is_equilibrium_property(self, name: str) -> bool:
        return isinstance(
            getattr(Equilibrium, name, None),
            property,
        )

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = str(value).lower()

        if self._mode not in self._SUPPORTED_MODES:
            raise ValueError("CombustionLookup mode must be 'hp' or 'tp'.")

    @property
    def equilibrium(self) -> Equilibrium:
        return self._Equilibrium

    @classmethod
    def supported_properties(cls) -> list[str]:
        return Equilibrium.supported_properties()

    @classmethod
    def show_supported_properties(cls) -> list[str]:
        return Equilibrium.show_supported_properties()

    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        return Equilibrium.supports_property(property_name)

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
            "pressure",
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
            "Equilibrium",
            "_Equilibrium",
            "input",
            "_input",
            "reactant_composition",
            "_reactant_composition",
            "last_flash_values",
            "_last_flash_values",
            "last_reactant_composition_values",
            "_last_reactant_composition_values",
            "property_cache",
            "_property_cache",
            "guess_temperature",
            "_guess_temperature",
            "candidates",
            "_candidates",
            "include_all_valid_gases",
            "_include_all_valid_gases",
            "element_tol",
            "_element_tol",
            "enthalpy_tol",
            "_enthalpy_tol",
            "correction_tol",
            "_correction_tol",
            "max_iterations",
            "_max_iterations",
            "trace_moles",
            "_trace_moles",
            "min_temperature",
            "_min_temperature",
            "max_temperature",
            "_max_temperature",
            "CombustionGas_trace",
            "_CombustionGas_trace",
            "CombustionGas_max_species",
            "_CombustionGas_max_species",
            "equilibrium_derivative_temperature_step",
            "_equilibrium_derivative_temperature_step",
        }