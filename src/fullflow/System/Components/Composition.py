from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fullflow.System import Component, State
from fullflow.System.State import is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Composition(Component):
    """
    Conserves arbitrary quantities carried by mass flow.

    `Composition` is a generic conservation component. It is not a
    ThermoProp composition object, and it does not know anything about
    chemistry, species, phases, or mixtures. The entries in the stream
    dictionaries are only user-defined labels such as "n2", "o2", "fuel",
    "liquid", "air", or any other quantity carried by mass flow.

    For each conserved label, one residual is created:

        sum(mdot_in * x_in) - sum(mdot_out * x_out) = 0

    where `x` is the value of that label in each stream. For mass-fraction-like
    quantities, `x` is usually a mass fraction. For other uses, `x` can be any
    user-defined transported value.

    Parameters
    ----------
    name : str
        Component name.

    network : Network
        Network that owns this component.

    inlets : list[tuple[State | float, dict[str, State | float] | None]]
        Incoming streams.

        Each inlet is written as:

            (mass_flow, composition)

        where `mass_flow` is the stream mass flow rate and `composition` maps
        labels to values carried by that stream.

        Example:

            (Feed.mass_flow, {"n2": 0.78, "o2": 0.21})

        Missing labels are treated as zero. Extra labels are ignored unless
        they are included in `names` or inferred from `solve`.

    outlets : list[tuple[State | float, dict[str, State | float] | None]]
        Outgoing streams.

        Each outlet is written as:

            (mass_flow, composition)

        where `mass_flow` is the stream mass flow rate and `composition` maps
        labels to values carried by that stream.

        Example:

            (Waste.mass_flow, {"n2": Waste.n2_fraction})

        Missing labels are treated as zero. Extra labels are ignored unless
        they are included in `names` or inferred from `solve`.

        If `composition` is None, the stream uses the corresponding value from
        `solve`. This is useful for well-mixed outlets.

        Example:

            outlets=[(Mixed.mass_flow, None)]
            solve={"n2": Mixer.n2_fraction}

        is equivalent to:

            outlets=[(Mixed.mass_flow, {"n2": Mixer.n2_fraction})]

    solve : dict[str, State] | None, optional
        States this component should iterate.

        Every State in `solve.values()` is returned as an iteration variable.
        The keys usually identify the labels being conserved.

        Example:

            solve={"n2": Waste.n2_fraction, "o2": Waste.o2_fraction}

        means the Composition component will iterate `Waste.n2_fraction` and
        `Waste.o2_fraction`.

        If `names` is omitted, the conserved labels are inferred from
        `solve.keys()`. Therefore, for most user examples, `solve` defines both
        the unknowns and the conservation labels.

    names : tuple[str, ...] | list[str] | str | None, optional
        Labels to conserve.

        Each label creates one residual. If omitted, labels are inferred from
        `solve.keys()`.

        Example:

            names=("n2", "o2")

        creates one residual for "n2" conservation and one residual for "o2"
        conservation.

        Use `names` only when you need conservation residuals whose labels are
        not exactly the same as the keys in `solve`, or when this component has
        no solved composition states but still needs to contribute residuals.

    mass : State | float | None, optional
        Placeholder for future transient composition storage.

        This is currently unused in steady-state calculations. Later, transient
        solvers can use it for equations like:

            d(m * x) / dt = sum(mdot_in * x_in) - sum(mdot_out * x_out)

    Notes
    -----
    `Composition` only writes conservation residuals. It does not normalize
    fractions, enforce bounds, or automatically require values to sum to one.
    This keeps the component generic enough for both mixture fractions and
    arbitrary transported quantities such as dye concentration, salt loading,
    liquid fraction, or a progress variable.

    For fraction-based mixtures, the recommended pattern is to solve only the
    independent fractions and define the final fraction as a dependent state.
    For an N-component mixture, this usually means conserving N - 1 labels if
    total mass conservation is already enforced elsewhere by a `Volume`.

    Example with N2/O2/Ar:

        Waste.n2_fraction = State(0.20)
        Waste.o2_fraction = State(0.75)
        Waste.ar_fraction = 1.0 - Waste.n2_fraction - Waste.o2_fraction

        Composition(
            "Separator Composition",
            network,
            inlets=[
                (Feed.mass_flow, {"n2": Feed.n2_fraction, "o2": Feed.o2_fraction}),
            ],
            outlets=[
                (Product.mass_flow, {"n2": 0.990, "o2": 0.005}),
                (Waste.mass_flow, {"n2": Waste.n2_fraction, "o2": Waste.o2_fraction}),
            ],
            solve={
                "n2": Waste.n2_fraction,
                "o2": Waste.o2_fraction,
            },
        )

    In that example, `Composition` enforces N2 and O2 conservation. The Volume
    enforces total mass conservation. Because argon is defined as the remaining
    fraction, argon conservation is implied without adding an independent argon
    residual.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        inlets: list[tuple[State | float, dict[str, State | float] | None]],
        outlets: list[tuple[State | float, dict[str, State | float] | None]],
        solve: dict[str, State | float] | None = None,
        names: tuple[str, ...] | list[str] | str | None = None,
        mass: State | float | None = None,
    ):
        self.setup()

        self.inlets.value = list(self.inlets.value)
        self.outlets.value = list(self.outlets.value)

        if self.solve.is_assigned:
            self.solve.value = self._normalize_solve(self.solve.value)
        else:
            self.solve.value = {}

        if self.names.is_assigned:
            self.names.value = self._normalize_names(self.names.value)
        else:
            self.names.value = tuple(self.solve.value)

        if not self.names.value:
            raise ValueError(
                f"{self.name}: no composition labels were provided. "
                "Pass names=(...) or solve={...}."
            )

    @staticmethod
    def _normalize_names(names: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
        if names is None:
            return ()

        if isinstance(names, str):
            return (names,)

        return tuple(str(name) for name in names)

    @staticmethod
    def _normalize_solve(solve: dict[str, State | float] | None) -> dict[str, State | float]:
        if solve is None:
            return {}

        return {
            str(name): value
            for name, value in solve.items()
        }

    @staticmethod
    def _value(value: State | float) -> Any:
        return value.value if is_state_like(value) else value

    def _composition_value(self, composition: dict[str, State | float] | None, name: str) -> float:
        if composition is None:
            if name not in self.solve.value:
                raise ValueError(
                    f"{self.name}: a stream uses None for {name!r}, "
                    f"but solve does not contain {name!r}."
                )

            return float(self._value(self.solve.value[name]))

        if name not in composition:
            return 0.0

        return float(self._value(composition[name]))

    def _composition_flow_rate(self, name: str) -> float:
        composition_flow_rate = 0.0

        for mass_flow, composition in self.inlets.value:
            composition_flow_rate += float(self._value(mass_flow)) * self._composition_value(composition, name)

        for mass_flow, composition in self.outlets.value:
            composition_flow_rate -= float(self._value(mass_flow)) * self._composition_value(composition, name)

        return composition_flow_rate

    @property
    def iteration_variables(self) -> list[State]:
        return [
            value
            for value in self.solve.value.values()
            if is_state_like(value)
        ]

    @property
    def residuals(self) -> list[State | float]:
        return [
            self._composition_flow_rate(name)
            for name in self.names.value
        ]
