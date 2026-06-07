from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State



class Thermocouple(Component):

    def __init__(self, 
                 name: str, 
                 network: Network,
                 target: State,
                 time_constant: float | None = None,
                 measurement: State | None = None):
        self.setup()

    def evaluate_states(self):
        
        self.measurement.value = self.target.value
