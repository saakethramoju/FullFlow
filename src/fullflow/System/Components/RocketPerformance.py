from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network



class SpecificImpulse(Component):

    """Specific impulse from thrust and propellant mass flow."""
    def __init__(
        self, 
        name: str, 
        network: Network,
        thrust: State,
        mass_flow: State,
        gravitational_acceleration: float = 9.80665,
        specific_impulse: State | None = None
        ):
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        self.specific_impulse.value = self.thrust.value / (self.mass_flow.value * self.gravitational_acceleration.value)





class IdealCharacteristicVelocity(Component):

    """Ideal-gas characteristic velocity diagnostic."""
    def __init__(
        self, 
        name: str, 
        network: Network,
        total_temperature: State,
        gas_constant: float,
        specific_heat_ratio: float,
        characteristic_velocity: State | None = None
        ):
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        g = self.specific_heat_ratio.value
        numerator = math.sqrt(g * self.gas_constant.value* self.total_temperature.value)
        denominator = g * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        self.characteristic_velocity.value = numerator / denominator
