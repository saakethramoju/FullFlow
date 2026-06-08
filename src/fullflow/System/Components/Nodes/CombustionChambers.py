from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class MainCombustionChamber(Component):
    """
    Lumped main combustion chamber mass-balance component.

    `MainCombustionChamber` solves for chamber pressure by enforcing steady-state
    mass conservation between the incoming propellant flow rates and the outgoing
    nozzle mass flow rate.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state chamber mass conservation:

        `fuel_mass_flow + oxidizer_mass_flow - nozzle_mass_flow = 0`

    Iteration Variables
    -------------------
    chamber_pressure : State
        Main combustion chamber pressure.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    chamber_pressure : State
        Main combustion chamber pressure [Pa].
    oxidizer_mass_flow : State, optional
        Oxidizer mass flow rate entering the chamber [kg/s].
    fuel_mass_flow : State, optional
        Fuel mass flow rate entering the chamber [kg/s].
    nozzle_mass_flow : State, optional
        Combustion gas mass flow rate leaving through the nozzle [kg/s].
    """

    def __init__(self, 
                 name: str, 
                 network: Network,
                 chamber_pressure: State,
                 oxidizer_mass_flow : State | None = None,
                 fuel_mass_flow: State | None = None, 
                 nozzle_mass_flow: State | None = None):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.chamber_pressure]

    @property
    def residuals(self) -> list[float]:
        return [self.fuel_mass_flow.value + self.oxidizer_mass_flow.value - self.nozzle_mass_flow.value]