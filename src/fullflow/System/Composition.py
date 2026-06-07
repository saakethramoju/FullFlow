import numpy as np

from thermoprop import FluidRegistry
from .State import State


class Composition:
    """
    Container for species mass-fraction States.
    """

    def __init__(
        self,
        fluid: dict[str, State | float] | str | None = None,
    ):
        self.fraction: dict[str, State] = {}
        self._constrained_species: str | None = None
        self._zero_fraction_states: dict[str, State] = {}

        if fluid is None:
            return

        if isinstance(fluid, str):
            fluid = {fluid: 1.0}

        self.fraction = {
            FluidRegistry.name(species): (
                value if isinstance(value, State) else State(float(value))
            )
            for species, value in fluid.items()
        }

        self.validate()

    def validate(self, atol: float = 1e-6) -> None:
        total = sum(state.value for state in self.fraction.values())

        if not np.isclose(total, 1.0, rtol=0.0, atol=atol):
            raise ValueError(
                f"Composition mass fractions must sum to 1.0. Got {total}."
            )

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.fraction.keys())

    @property
    def values(self) -> dict[str, float]:
        return {
            species: state.value
            for species, state in self.fraction.items()
        }

    @property
    def is_assigned(self) -> bool:
        return len(self.fraction) > 0

    def update(self) -> None:
        if self._constrained_species is None:
            self.constrain_species()

        self.enforce_constraint()

    def constrain_species(self, species: str | None = None) -> None:
        if not self.is_assigned:
            return

        if self._constrained_species is not None and species is None:
            return

        if species is None:
            species = next(reversed(self.fraction))

        species = FluidRegistry.name(species)

        if species not in self.fraction:
            raise ValueError(f"{species!r} is not present in the composition.")

        self._constrained_species = species
        self.enforce_constraint()

    def enforce_constraint(self) -> None:
        if self._constrained_species is None:
            return

        self.fraction[self._constrained_species].value = 1.0 - sum(
            state.value
            for species, state in self.fraction.items()
            if species != self._constrained_species
        )


    def copy_from(
        self,
        other: "Composition",
        copy_values: bool = True,
    ) -> None:
        for species in other.species:
            if species in self.fraction:
                if copy_values:
                    self.fraction[species].value = other[species].value
            else:
                value = other[species].value if copy_values else 0.0
                self.fraction[species] = State(value)

        self._zero_fraction_states.clear()

        if copy_values:
            if self._constrained_species is None and self.is_assigned:
                self.constrain_species()
            else:
                self.enforce_constraint()
        

    def __getitem__(self, species: str) -> State:
        species = FluidRegistry.name(species)

        if species in self.fraction:
            return self.fraction[species]

        if species not in self._zero_fraction_states:
            self._zero_fraction_states[species] = State(0.0)

        return self._zero_fraction_states[species]


    def __and__(self, other: "Composition | None") -> tuple[str, ...]:
        """
        Returns the species intersection between two compositions.
        """

        if other is None or not self.is_assigned or not other.is_assigned:
            return ()

        return tuple(
            species
            for species in self.species
            if species in other
        )

    def __iter__(self):
        return iter(self.fraction.items())

    def __contains__(self, species: str) -> bool:
        return FluidRegistry.name(species) in self.fraction

    def __len__(self) -> int:
        return len(self.fraction)

    def __str__(self) -> str:
        if not self.is_assigned:
            return "Composition(<unassigned>)"

        return (
            "Composition("
            + ", ".join(
                f"{species}={state.value:.6g}"
                for species, state in self.fraction.items()
            )
            + ")"
        )

    def __repr__(self) -> str:
        return f"Composition({self.values})"