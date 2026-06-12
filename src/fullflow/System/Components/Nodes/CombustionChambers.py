from __future__ import annotations

from thermoprop import Propellant, Reactants
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network

class CombustionChamber(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        chamber_pressure: State,
        fuel: str,
        oxidizer: str,
        fuel_temperature: State | float,
        oxidizer_temperature: State | float,
        fuel_mass_flow: State | None = None,
        oxidizer_mass_flow: State | None = None,
        mass_flow_out: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        Pc = self.chamber_pressure.value
        fuel_mdot = abs(self.fuel_mass_flow.value)
        ox_mdot = abs(self.oxidizer_mass_flow.value)
        MR = max(ox_mdot/fuel_mdot, 0.1)
        try:
            fuelProp = Propellant(self.fuel, temperature=self.fuel_temperature.value, pressure=Pc)
        except:
            fuelProp = Propellant(self.fuel, temperature=self.fuel_temperature.value)

        try:
            oxProp = Propellant(self.oxidizer, temperature=self.oxidizer_temperature.value, pressure=Pc)
        except:
            oxProp = Propellant(self.oxidizer, temperature=self.oxidizer_temperature.value)

        reactants = Reactants(
            fuels=self.fuel,
            oxidizers=self.oxidizer,
            mixture_ratio=MR
        )

        






class MainCombustionChamber(Component):
    """
    Main combustion chamber mass-balance node.

    `MainCombustionChamber` represents a lumped combustion chamber whose pressure
    is solved from steady-state mass conservation. The chamber pressure adjusts
    such that the total incoming propellant mass flow matches the mass flow
    leaving through the nozzle.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state chamber mass conservation

        ``fuel_mass_flow
        + oxidizer_mass_flow
        - nozzle_mass_flow
        = 0``

    Iteration Variables
    -------------------
    chamber_pressure : State
        Main combustion chamber pressure

    Parameters
    ----------
    name : str
        Component name

    network : Network
        Network that owns this component

    chamber_pressure : State
        Main combustion chamber pressure

    oxidizer_mass_flow : State, optional
        Oxidizer mass flow entering the chamber

    fuel_mass_flow : State, optional
        Fuel mass flow entering the chamber

    nozzle_mass_flow : State, optional
        Combustion gas mass flow leaving through the nozzle
    """

    def __init__(
        self,
        name: str,
        network: Network,
        chamber_pressure: State,
        oxidizer_mass_flow: State | None = None,
        fuel_mass_flow: State | None = None,
        nozzle_mass_flow: State | None = None,
    ):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.chamber_pressure]

    @property
    def residuals(self) -> list[float]:
        return [
            self.fuel_mass_flow.value
            + self.oxidizer_mass_flow.value
            - self.nozzle_mass_flow.value
        ]