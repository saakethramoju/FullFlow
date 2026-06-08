from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State, Composition

if TYPE_CHECKING:
    from fullflow.System import Network


class FlowMixer(Component):
    """
    Two-inlet mixing volume with optional energy and species conservation.

    `FlowMixer` combines two incoming streams into a single outlet stream while
    enforcing steady-state mass conservation. Optional energy and species
    balances may also be solved when the corresponding states are provided.

    The outlet composition is calculated from the mass-flow-weighted mixture of
    the two inlet compositions.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state mass conservation:

        `mass_flow_in1 + mass_flow_in2 - mass_flow_out = 0`

    energy_balance : float, optional
        Enforces steady-state flow energy conservation:

        `mass_flow_in1 * total_enthalpy_in1
        + mass_flow_in2 * total_enthalpy_in2
        - mass_flow_out * total_enthalpy_out
        + heat_rate = 0`

        Included only when energy solving is enabled.

    Relations
    ---------
    Species mixing:

    `Y_out = (mdot1 * Y1 + mdot2 * Y2) / (mdot1 + mdot2)`

    for each species present in either inlet stream.

    Iteration Variables
    -------------------
    pressure : State
        Mixer pressure.

    enthalpy : State, optional
        Mixer specific enthalpy when energy conservation is enabled.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Mixer pressure [Pa].
    volume : float
        Mixer control volume [m^3].
    enthalpy : State, optional
        Mixer specific enthalpy [J/kg].
    temperature : State, optional
        Mixer temperature [K].
    density : State, optional
        Mixer density [kg/m^3].
    internal_energy : State, optional
        Mixer specific internal energy [J/kg].
    heat_rate : State or float, optional
        Net heat rate into the mixer [W].
    composition : Composition, optional
        Mixed outlet composition.
    composition_in1 : Composition, optional
        Composition of inlet stream 1.
    composition_in2 : Composition, optional
        Composition of inlet stream 2.
    total_enthalpy_in1 : State, optional
        Total specific enthalpy of inlet stream 1 [J/kg].
    total_enthalpy_in2 : State, optional
        Total specific enthalpy of inlet stream 2 [J/kg].
    total_enthalpy_out : State, optional
        Outlet total specific enthalpy [J/kg]. If omitted, the mixer enthalpy
        is used.
    mass_flow_in1 : State, optional
        Inlet mass flow rate 1 [kg/s].
    mass_flow_in2 : State, optional
        Inlet mass flow rate 2 [kg/s].
    mass_flow_out : State, optional
        Outlet mass flow rate [kg/s].
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: float,
        enthalpy: State | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        heat_rate: State | float | None = None,
        composition: Composition = Composition(),
        composition_in1: Composition = Composition(),
        composition_in2: Composition = Composition(),
        total_enthalpy_in1: State | None = None,
        total_enthalpy_in2: State | None = None,
        total_enthalpy_out: State | None = None,
        mass_flow_in1: State | None = None,
        mass_flow_in2: State | None = None,
        mass_flow_out: State | None = None,
    ):
        self.setup()

        self._solve_energy = (
            self.total_enthalpy_in1.is_assigned
            and self.enthalpy.is_assigned
            and self.total_enthalpy_in2.is_assigned
        )

        self._solve_species = (
            self.composition.is_assigned
            and self.composition_in1.is_assigned
            and self.composition_in2.is_assigned
        )

    def evaluate_states(self):
        if not self._solve_species:
            return

        extra_species = set(self.composition.species) - (
            set(self.composition_in1.species)
            | set(self.composition_in2.species)
        )

        if extra_species:
            raise ValueError(
                f"{self.name}: composition contains species not in "
                f"composition_in1 or composition_in2: {extra_species}"
            )

        self.composition.copy_from(self.composition_in1, copy_values=False)
        self.composition.copy_from(self.composition_in2, copy_values=False)

        if (
            not self.mass_flow_in1.is_assigned
            or not self.mass_flow_in2.is_assigned
            or not self.mass_flow_out.is_assigned
        ):
            return

        mdot1 = self.mass_flow_in1.value
        mdot2 = self.mass_flow_in2.value
        mdot_total = mdot1 + mdot2

        if abs(mdot_total) < 1e-12:
            return

        for species in self.composition.species:
            yi1 = (
                self.composition_in1[species].value
                if species in self.composition_in1.species
                else 0.0
            )

            yi2 = (
                self.composition_in2[species].value
                if species in self.composition_in2.species
                else 0.0
            )

            self.composition[species].value = (
                mdot1 * yi1 + mdot2 * yi2
            ) / mdot_total

    @property
    def iteration_variables(self) -> list[State]:
        variables = [self.pressure]

        if self._solve_energy:
            variables.append(self.enthalpy)

        return variables

    @property
    def residuals(self) -> list[float]:
        residuals = [
            self.mass_flow_in1.value
            + self.mass_flow_in2.value
            - self.mass_flow_out.value
        ]

        if self._solve_energy:
            qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0

            h_out = (
                self.total_enthalpy_out.value
                if self.total_enthalpy_out.is_assigned
                else self.enthalpy.value
            )

            residuals.append(
                self.mass_flow_in1.value * self.total_enthalpy_in1.value
                + self.mass_flow_in2.value * self.total_enthalpy_in2.value
                - self.mass_flow_out.value * h_out
                + qdot
            )

        return residuals
    

'''
class FlowMixer(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: float,
        enthalpy: State | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        heat_rate: State | float | None = None,
        composition: Composition = Composition(),
        composition_in1: Composition = Composition(),
        composition_in2: Composition = Composition(),
        total_enthalpy_in1: State | None = None,
        total_enthalpy_in2: State | None = None,
        total_enthalpy_out: State | None = None,
        mass_flow_in1: State | None = None,
        mass_flow_in2: State | None = None,
        mass_flow_out: State | None = None,
    ):
        self.setup()

        self._solve_energy = (
            self.total_enthalpy_in1.is_assigned
            and self.total_enthalpy_in2.is_assigned
            and self.enthalpy.is_assigned
        )

        self._solve_species = (
            self.composition.is_assigned
            and self.composition_in1.is_assigned
            and self.composition_in2.is_assigned
        )

    def evaluate_states(self) -> None:
        if not (
            self.mass_flow_in1.is_assigned
            and self.mass_flow_in2.is_assigned
            and self.mass_flow_out.is_assigned
        ):
            return

        mdot1 = self.mass_flow_in1.value
        mdot2 = self.mass_flow_in2.value
        mdot_total = mdot1 + mdot2

        if abs(mdot_total) < 1e-12:
            return

        if self._solve_species:
            inlet_species = (
                set(self.composition_in1.species)
                | set(self.composition_in2.species)
            )

            extra_species = set(self.composition.species) - inlet_species

            if extra_species:
                raise ValueError(
                    f"{self.name}: composition contains species not in "
                    f"composition_in1 or composition_in2: {extra_species}"
                )

            self.composition.copy_from(self.composition_in1, copy_values=False)
            self.composition.copy_from(self.composition_in2, copy_values=False)

            for species in self.composition.species:
                yi1 = (
                    self.composition_in1[species].value
                    if species in self.composition_in1.species
                    else 0.0
                )

                yi2 = (
                    self.composition_in2[species].value
                    if species in self.composition_in2.species
                    else 0.0
                )

                self.composition[species].value = (
                    mdot1 * yi1 + mdot2 * yi2
                ) / mdot_total

            self.composition.validate()

        if self._solve_energy:
            qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0

            self.enthalpy.value = (
                mdot1 * self.total_enthalpy_in1.value
                + mdot2 * self.total_enthalpy_in2.value
                + qdot
            ) / mdot_total

    @property
    def iteration_variables(self) -> list[State]:
        return [self.pressure]

    @property
    def residuals(self) -> list[float]:
        return [
            self.mass_flow_in1.value
            + self.mass_flow_in2.value
            - self.mass_flow_out.value
        ]
'''




class FlowSplitter(Component):
    """
    One-inlet, two-outlet flow splitter with optional energy and species conservation.

    `FlowSplitter` divides a single incoming stream into two outlet streams while
    enforcing steady-state mass conservation. Optional energy conservation may be
    solved, and optional species conservation can be used to determine the
    composition of the second outlet stream.

    When species conservation is enabled, the composition of outlet 1 is treated
    as known and the composition of outlet 2 is calculated from species mass
    balances.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state mass conservation:

        `mass_flow_in
        - mass_flow_out1
        - mass_flow_out2 = 0`

    energy_balance : float, optional
        Enforces steady-state flow energy conservation:

        `mass_flow_in * total_enthalpy_in
        - mass_flow_out1 * total_enthalpy_out1
        - mass_flow_out2 * total_enthalpy_out2
        + heat_rate = 0`

        Included only when energy solving is enabled.

    Relations
    ---------
    Species conservation for outlet 2:

    `Y_out2 =
    ((mdot_out1 + mdot_out2) * Y_internal
    - mdot_out1 * Y_out1)
    / mdot_out2`

    for each species in the splitter composition.

    Notes
    -----
    Species conservation solves only for the composition of outlet 2. The
    composition of outlet 1 must be specified.

    Iteration Variables
    -------------------
    pressure : State
        Splitter pressure.

    enthalpy : State, optional
        Splitter specific enthalpy when energy conservation is enabled.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Splitter pressure [Pa].
    volume : float
        Splitter control volume [m^3].
    enthalpy : State, optional
        Splitter specific enthalpy [J/kg].
    temperature : State, optional
        Splitter temperature [K].
    density : State, optional
        Splitter density [kg/m^3].
    internal_energy : State, optional
        Splitter specific internal energy [J/kg].
    heat_rate : State or float, optional
        Net heat rate into the splitter [W].
    composition : Composition, optional
        Internal splitter composition.
    composition_in : Composition, optional
        Inlet composition.
    composition_out1 : Composition, optional
        Outlet 1 composition.
    composition_out2 : Composition, optional
        Outlet 2 composition.
    total_enthalpy_in : State, optional
        Inlet total specific enthalpy [J/kg].
    total_enthalpy_out1 : State, optional
        Outlet 1 total specific enthalpy [J/kg]. If omitted, the splitter
        enthalpy is used.
    total_enthalpy_out2 : State, optional
        Outlet 2 total specific enthalpy [J/kg]. If omitted, the splitter
        enthalpy is used.
    mass_flow_in : State, optional
        Inlet mass flow rate [kg/s].
    mass_flow_out1 : State, optional
        Outlet mass flow rate 1 [kg/s].
    mass_flow_out2 : State, optional
        Outlet mass flow rate 2 [kg/s].
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: float,
        enthalpy: State | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        heat_rate: State | float | None = None,
        composition: Composition = Composition(),
        composition_in: Composition = Composition(),
        composition_out1: Composition = Composition(),
        composition_out2: Composition = Composition(),
        total_enthalpy_in: State | None = None,
        total_enthalpy_out1: State | None = None,
        total_enthalpy_out2: State | None = None,
        mass_flow_in: State | None = None,
        mass_flow_out1: State | None = None,
        mass_flow_out2: State | None = None,
    ):
        self.setup()

        self._solve_energy = (
            self.total_enthalpy_in.is_assigned
            and self.enthalpy.is_assigned
        )

        if self.composition_in.is_assigned:
            self.composition.copy_from(
                self.composition_in,
                copy_values=True,
            )

        self._solve_species = (
            self.composition.is_assigned
            and self.composition_out1.is_assigned
        )

        if self._solve_species:
            extra_species = (
                set(self.composition_out1.species)
                - set(self.composition.species)
            )

            if extra_species:
                raise ValueError(
                    f"{self.name}: composition_out1 contains species not in "
                    f"composition: {extra_species}"
                )

            self.composition_out2.copy_from(
                self.composition,
                copy_values=True,
            )

        elif self.composition.is_assigned:
            self.composition_out1.copy_from(
                self.composition,
                copy_values=True,
            )

            self.composition_out2.copy_from(
                self.composition,
                copy_values=True,
            )

    def evaluate_states(self) -> None:
        if self.composition_in.is_assigned:
            self.composition.copy_from(
                self.composition_in,
                copy_values=True,
            )

        if not self._solve_species:
            if self.composition.is_assigned:
                self.composition_out1.copy_from(
                    self.composition,
                    copy_values=True,
                )
                self.composition_out2.copy_from(
                    self.composition,
                    copy_values=True,
                )
            return

        if (
            not self.mass_flow_out1.is_assigned
            or not self.mass_flow_out2.is_assigned
        ):
            return

        mdot1 = self.mass_flow_out1.value
        mdot2 = self.mass_flow_out2.value

        if abs(mdot2) < 1e-12:
            return

        mdot_total = mdot1 + mdot2

        for species in self.composition.species:
            x_internal = self.composition[species].value
            x_out1 = self.composition_out1[species].value

            self.composition_out2[species].value = (
                mdot_total * x_internal
                - mdot1 * x_out1
            ) / mdot2

        self.composition_out2.validate()

    @property
    def iteration_variables(self) -> list[State]:
        variables = [self.pressure]

        if self._solve_energy:
            variables.append(self.enthalpy)

        return variables

    @property
    def residuals(self) -> list[float]:
        residuals = [
            self.mass_flow_in.value
            - self.mass_flow_out1.value
            - self.mass_flow_out2.value
        ]

        if self._solve_energy:
            qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0

            h_out1 = (
                self.total_enthalpy_out1.value
                if self.total_enthalpy_out1.is_assigned
                else self.enthalpy.value
            )

            h_out2 = (
                self.total_enthalpy_out2.value
                if self.total_enthalpy_out2.is_assigned
                else self.enthalpy.value
            )

            residuals.append(
                self.mass_flow_in.value * self.total_enthalpy_in.value
                - self.mass_flow_out1.value * h_out1
                - self.mass_flow_out2.value * h_out2
                + qdot
            )

        return residuals