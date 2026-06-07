from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class MainCombustionChamber(Component):

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