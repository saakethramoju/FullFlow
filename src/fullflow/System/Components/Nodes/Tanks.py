from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class SimpleIsothermalTank(Component):

    def __init__(self, 
                 name: str,
                 network: Network,
                 pressure: State,
                 temperature: State | None = None,
                 liquid_height: State | None = None,
                 gravitation_acceleration: State | None = 9.80665,
                 head_pressure: State | None = None,
                 internal_energy: State | None = None,
                 density: State | None = None,
                 mass_flow_out: State | None = None):
        self.setup()


    def evaluate_states(self):
        if self.liquid_height.is_assigned:
            self.head_pressure.value = (self.density.value 
                                        * self.gravitational_acceleration.value 
                                        * self.liquid_height.value)
            
            


class PressurizedTank(Component):

    def __init__(self, 
                 name: str, 
                 network: Network, 
                 pressure: State, 
                 pressurant_density: State, 
                 liquid_density: State, 
                 collapse_factor: State | float | None = 1, 
                 ullage_temperature: float | None = None, 
                 liquid_temperature: float | None = None, 
                 liquid_height: State | None = None, 
                 gravitation_acceleration: State | None = 9.80665, 
                 head_pressure: State | None = None, 
                 pressurant_internal_energy: State | None = None, 
                 liquid_internal_energy: State | None = None,
                 liquid_enthalpy: State | None = None,
                 mass_flow_in: State | None = None,
                 mass_flow_out: State | None = None):
        self.setup()

    def evaluate_states(self):
        if self.liquid_height.is_assigned:
            self.head_pressure.value = (self.density.value 
                                        * self.gravitational_acceleration.value 
                                        * self.liquid_height.value)


    @property
    def iteration_variables(self) -> list[State]:
        return [self.pressure]
    
    @property
    def residuals(self) -> list[float]:
        return [self.mass_flow_in.value 
                - self.collapse_factor.value * self.pressurant_density.value
                * self.mass_flow_out.value / self.liquid_density.value]